/*
 * avenir: Predictive analytic based on Hadoop Map Reduce
 * Author: Pranab Ghosh
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You may
 * obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0 
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */


package org.avenir.tree;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.tree.DecisionPathList.DecisionPathPredicate;
import org.avenir.tree.SplitManager.AttributePredicate;
import org.avenir.util.AttributeSplitStat;
import org.avenir.util.InfoContentStat;
import org.chombo.util.BasicUtils;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Pair;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * @author pranab
 *
 */
public class DecisionTreeBuilder   extends Configured implements Tool {
    public static final String ROOT_PATH = "$root";
    private static final String CHILD_PATH = "$child";
    //public static final String PRED_DELIM = ";";
    public static final String SPLIT_DELIM = ":";
    private static final Logger LOG = Logger.getLogger(DecisionTreeBuilder.class);

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Decision tree  builder";
        job.setJobName(jobName);
        job.setJarByClass(DecisionTreeBuilder.class);
        Utility.setConfiguration(job.getConfiguration(), "avenir");

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
       
        job.setMapperClass(DecisionTreeBuilder.BuilderMapper.class);
        job.setReducerClass(DecisionTreeBuilder.BuilderReducer.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("dtb.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * Decision tree or random forest. For random forest, data is sampled in the first iteration
	 * @author pranab
	 *
	 */
	public static class BuilderMapper extends Mapper<LongWritable, Text, Tuple, Text> {
		private String fieldDelimRegex;
		private String[] items;
        private Tuple outKey = new Tuple();
		private Text outVal  = new Text();
        private FeatureSchema schema;
        private List<Integer> splitAttrs;
        private FeatureField classField;
        private SplitManager splitManager;
        private String attrSelectStrategy;
        private int randomSplitSetSize;
        private String classVal;
        private String currenttDecPath;
        private String decPathDelim;
        private DecisionPathList decPathList;
        private Map<String, Boolean> validDecPaths = new HashMap<String, Boolean>();
        private String subSamlingStrategy;
        private boolean treeAvailable;
        private int samplingRate;
        private int samplingBufferSize;
        private String[]  samplingBuffer;
        private int count;
        private boolean debugOn;
        private static final String SUB_SAMPLING_WITH_REPLACE = "withReplace";
        private static final String SUB_SAMPLING_WITHOUT_REPLACE = "withoutReplace";
        private static final String SUB_SAMPLING_WITHOUT_NONE = "none";
        private static final String ATTR_SEL_ALL = "all";
        private static final String ATTR_SEL_NOT_USED_YET = "notUsedYet";
        private static final String ATTR_SEL_RANDOM_ALL = "randomAll";
        private static final String ATTR_SEL_RANDOM_NOT_USED_YET = "randomNotUsedYet";
       

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	debugOn = conf.getBoolean("debug.on", false);
            if (debugOn) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	
        	//schema
            schema = Utility.getFeatureSchema(conf, "dtb.feature.schema.file.path");
            
            //decision path list  file
            InputStream  fs = Utility.getFileStream(context.getConfiguration(), "dtb.decision.file.path.in");
            if (null != fs) {
            	ObjectMapper  mapper = new ObjectMapper();
            	decPathList = mapper.readValue(fs, DecisionPathList.class);
            	treeAvailable = true;
            }
           
            //split manager
            decPathDelim = conf.get("dtb.dec.path.delim", ";");
            splitManager = new SplitManager(schema, decPathDelim); 
            splitManager.setDebugOn(debugOn);
            String customBaseAttributeOrdinalsStr = conf.get("dtb.custom.base.attributes");
            
            //use limited set of candidate attributes instead of all
            if (null != customBaseAttributeOrdinalsStr) {
            	int[] customBaseAttributeOrdinals = Utility.intArrayFromString(customBaseAttributeOrdinalsStr);
            	splitManager.withCustomBaseAttributeOrdinals(customBaseAttributeOrdinals);
            }
            
            //attribute selection strategy
            attrSelectStrategy = conf.get("dtb.split.attribute.selection.strategy", "notUsedYet");
 
           	randomSplitSetSize = conf.getInt("dtb.random.split.set.size", 3);
            
            //class attribute
            classField = schema.findClassAttrField();
            
            validDecPaths.clear();
            
            //sub sampling
            subSamlingStrategy = conf.get("dtb.sub.sampling.strategy", "withReplace");
            if (subSamlingStrategy.equals(SUB_SAMPLING_WITHOUT_REPLACE)) {
            	samplingRate = Utility.assertIntConfigParam(conf, "dtb.sub.sampling.rate", 
            			"samling rate should be provided for sampling without replacement");
            } else if (subSamlingStrategy.equals(SUB_SAMPLING_WITH_REPLACE)) {
            	int samplingBufferSize = conf.getInt("dtb.sub.sampling.buffer.size",  10000);
            	samplingBuffer = new String[samplingBufferSize];
            }
            
            //validate field ordinals
            int recLen = conf.getInt("dtb.rec.len", -1);
            if (recLen > 0) {
            	schema.validateFieldOrdinals(recLen);
            }
        }
        
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			if (!treeAvailable && subSamlingStrategy.equals(SUB_SAMPLING_WITH_REPLACE)) {
				//remaining in buffer
				for (int i = 0; i < count; ++i) {
					int sel = (int)(Math.random() * count);
					sel = sel == count ? count -1 : sel;
					rootMapHelper(samplingBuffer[sel], context);
				}
			}
			super.cleanup(context);
		}
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        	//sampling
        	if (!treeAvailable) {
        		//first iteration for root predicate
        		 if (subSamlingStrategy.equals(SUB_SAMPLING_WITH_REPLACE)) {
        			 	//sampling with replace
		        		if (count <  samplingBufferSize)  {
		        			samplingBuffer[count++] = value.toString();
		        		} else {
		        			//sample all
		        			for (int i = 0; i < samplingBufferSize; ++i) {
		        				int sel = (int)(Math.random() * samplingBufferSize);
		        				sel = sel == samplingBufferSize ? samplingBufferSize -1 : sel;
		        				rootMapHelper(samplingBuffer[sel], context);
		        			}
		        			
		        			//start refilling buffer
		        			count = 0;
		        			samplingBuffer[count++] = value.toString();
		        		}
        		 } else if (subSamlingStrategy.equals(SUB_SAMPLING_WITHOUT_REPLACE)) {
        			 	//sampling without replace
     					int sel = (int)(Math.random() * 100);
     					if (sel < samplingRate) {
     						rootMapHelper(value.toString(),  context);
     					}
        		 } else {
        			 //no sampling
        			 rootMapHelper(value.toString(),  context);
        		 }
        	} else {
        		//intermediate iteration
        		pathMapHelper(value.toString(),  context);
        	}
  		}

        /**
         * @param record
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void rootMapHelper(String record, Context context)
                throws IOException, InterruptedException {
			outKey.initialize();
			outKey.add(ROOT_PATH);
			outVal.set(record);
			context.write(outKey, outVal);
        }        
        
        /**
         * @param record
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void pathMapHelper(String record, Context context)
                throws IOException, InterruptedException {
       		items  =  record.split(fieldDelimRegex, -1);
       	    classVal = items[classField.getOrdinal()];
            	
             currenttDecPath = null;
              if (treeAvailable) {
            	    //strip split ID
                	currenttDecPath = items[0];
            		String[] predicates = DecisionPathList.stripSplitId(currenttDecPath.split(decPathDelim));
            		currenttDecPath = BasicUtils.join(predicates, decPathDelim);
            		
                	//find decision path status
                	Boolean status = validDecPaths.get(currenttDecPath);
                	if (null == status) {
                		//from decision path list object
                		DecisionPathList.DecisionPath decPathObj = decPathList.findDecisionPath(predicates) ;
                		status = null == decPathObj ? false : true;
                		
                		//cache it
                		validDecPaths.put(currenttDecPath, status);
                	} 
                	
                	//rejected decision path from earlier iteration rejected splits
                	if (!status) {
                		return;
                	}
                }
                
              //get split attributes
              getSplitAttributes();
                
              //all attributes
          	  int splitId = 0;
              for (int attr :  splitAttrs) {
        	  	if (attr > items.length-2) {
        	  		throw new IllegalStateException("attrbite index out of bound attr:" + attr + " rec length:" + (items.length-1));
        	  	}
        	  	
            	FeatureField field = schema. findFieldByOrdinal(attr);
            	Object attrValue = null;
            	//all splits, first field is the decision path, so all fields shifted by 1
            	List<List<AttributePredicate>> allSplitPredicates = null;
            	if (field.isInteger()) {
            		allSplitPredicates = splitManager.createIntAttrSplitPredicates(attr);
            		Integer iValue = Integer.parseInt(items[attr + 1]);
            		attrValue = iValue;
            	} else if (field.isDouble()) {
            		allSplitPredicates = splitManager.createDoubleAttrSplitPredicates(attr);
            		Double dValue = Double.parseDouble(items[attr + 1]);
            		attrValue = dValue;
            	} else if (field.isCategorical()) {
            		allSplitPredicates = splitManager.createCategoricalAttrSplitPredicates(attr);
            		attrValue = items[attr + 1];
            	}
                
            	//evaluate split predicates
                for (List<AttributePredicate> predicates : allSplitPredicates) {
                	//unique split id for each partion in a split
                	++splitId;
                	
                	//predicates for a split
                	boolean predicateMatched = false;
                	for (AttributePredicate predicate : predicates) {
                		if (predicate.evaluate(attrValue)) {
                			//data belongs to this split segment
                			predicateMatched = true;
                			outKey.initialize();
                			if (null == currenttDecPath) {
                				outKey.add(predicate.toString());
                    			outVal.set(record);
                			} else {
                				//existing predicates
                				String[] curDecPathItems = items[0].split(decPathDelim);
                				for (String curDecPathItem : curDecPathItems) {
                    				outKey.add(curDecPathItem);
                				}
                				
                				//new predicate
                 				outKey.add("" + splitId + SPLIT_DELIM +   predicate.toString());
                 				int pos = record.indexOf(fieldDelimRegex);
                 				
                 				//exclude predicate
                    			outVal.set(record.substring(pos + fieldDelimRegex.length()));
                			}               		
            				context.write(outKey, outVal);
                		}	
                	}
                	if (!predicateMatched) {
                		throw new IllegalStateException("no matching predicate for attribute: " + attr);
                	}
                }
              }
      	}
       
        /**
         * @param attrSelectStrategy
         * @param conf
         */
        private void getSplitAttributes() {
            if (attrSelectStrategy.equals(ATTR_SEL_ALL)) {
            	//all attributes
            	splitAttrs = splitManager.getAllAttributes();
            } else if (attrSelectStrategy.equals(ATTR_SEL_NOT_USED_YET)) {
            	//attributes that have not been used yet
            	splitAttrs = splitManager.getRemainingAttributes(currenttDecPath);
            } else if (attrSelectStrategy.equals(ATTR_SEL_RANDOM_ALL)) {
            	//randomly selected k attributes from all
            	splitManager.getRandomAllAttributes(randomSplitSetSize);
            } else if (attrSelectStrategy.equals(ATTR_SEL_RANDOM_NOT_USED_YET)) {
            	//randomly selected k attributes from attributes not used yet
            	splitManager.getRandomRemainingAttributes(currenttDecPath, randomSplitSetSize);
            } else {
            	throw new IllegalArgumentException("invalid splitting attribute selection strategy");
            }
        }	
	}

	/**
	 * @author pranab
	 *
	 */
	public static class BuilderReducer extends Reducer<Tuple, Text, NullWritable, Text> {
 		private FeatureSchema schema;
		private String fieldDelim;
		private Text outVal  = new Text();
		private String  infoAlgorithm;
        private boolean outputSplitProb;
        private Map<String, Map<String, InfoContentStat>> decPaths = new HashMap<String, Map<String, InfoContentStat>>();
        private Map<String, Map<String, Map<String, InfoContentStat>>> decPathsInfoContentBySplit = 
        		new HashMap<String, Map<String, Map<String, InfoContentStat>>>();
        private int classAttrOrdinal;
        private String classAttrValue;
        private String parentDecPath;
        private String decPath;
        private String childPath;
        private String decPathDelim;
        private DecisionPathStoppingStrategy pathStoppingStrategy;
        private DecisionPathList decPathList;
        private boolean decTreeAvailable;
        private String spltSelStrategy;
        private int topSplitCount;
        private int totalPopulation;
        private boolean debugOn;
        private static String  SPLIT_SEL_BEST = "best";
        private static String  SPLIT_SEL_RANDOM_TOP = "randomAmongTop";
        
	   	@Override
	   	protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	debugOn = conf.getBoolean("debug.on", false);
            if (debugOn) {
            	LOG.setLevel(Level.DEBUG);
            	AttributeSplitStat.enableLog();
            }
            
            //schema
            schema = Utility.getFeatureSchema(conf, "dtb.feature.schema.file.path");

            //decision path list  file
            InputStream fs = Utility.getFileStream(context.getConfiguration(), "dtb.decision.file.path.in");
            if (null != fs) {
            	ObjectMapper mapper = new ObjectMapper();
            	decPathList = mapper.readValue(fs, DecisionPathList.class);
            	decTreeAvailable = true;
            }
            
        	fieldDelim = conf.get("field.delim.out", ",");
        	infoAlgorithm = conf.get("dtb.split.algorithm", "giniIndex");
        	outputSplitProb = conf.getBoolean("dtb.output.split.prob", false);
        	classAttrOrdinal = schema.findClassAttrField().getOrdinal();
            decPathDelim = conf.get("dtb.dec.path.delim", ";");
            SplitManager.setPredDelim(decPathDelim);
            totalPopulation = conf.getInt("dtb.total.population", -1);
            
            //split selection strategy
            spltSelStrategy = conf.get("dtb.split.select.strategy", SPLIT_SEL_BEST);
            if (spltSelStrategy.equals(SPLIT_SEL_RANDOM_TOP)) {
            	topSplitCount = conf.getInt("dtb.top.split.count", 3);
            }

        	//stopping strategy
        	String stoppingStrategy =  conf.get("dtb.path.stopping.strategy", DecisionPathStoppingStrategy.STOP_MIN_INFO_GAIN);
        	int maxDepthLimit = -1;
        	double minInfoGainLimit = -1;
        	int minPopulationLimit = -1;
        	if (stoppingStrategy.equals(DecisionPathStoppingStrategy.STOP_MAX_DEPTH)) {
        		maxDepthLimit = Utility.assertIntConfigParam(conf, "dtb.max.depth.limit", "missing max depth limit for tree");
        	} else if (stoppingStrategy.equals(DecisionPathStoppingStrategy.STOP_MIN_INFO_GAIN)) {
            	minInfoGainLimit =  Utility.assertDoubleConfigParam(conf, "dtb.min.info.gain.limit", "missing min info gain limit");     
        	} else if (stoppingStrategy.equals(DecisionPathStoppingStrategy.STOP_MIN_POPULATION)) {
            	minPopulationLimit =  Utility.assertIntConfigParam(conf, "dtb.min.population.limit", "missing min population limit");                 
        	} else {
        		throw new IllegalArgumentException("invalid stopping strategy " + stoppingStrategy);
        	}
        	pathStoppingStrategy = new DecisionPathStoppingStrategy(stoppingStrategy, maxDepthLimit, 
        			minInfoGainLimit,minPopulationLimit);
	   	}   

	   	@Override
	   	protected void cleanup(Context context)  throws IOException, InterruptedException {
	   		if (decTreeAvailable) {
	   			expandTree(context);
	   		} else {
	   			generateRoot(context);
	   		}
	   	}
	   	
	   	/**
	   	 * @throws IOException 
	   	 * 
	   	 */
	   	private void generateRoot(Context context) throws IOException {
	   		boolean isAlgoEntropy = infoAlgorithm.equals("entropy");
	   		Map<String, InfoContentStat> childStats = decPaths.get(ROOT_PATH);
	   		InfoContentStat childStat = childStats.get(CHILD_PATH);
	   		childStat.processStat(isAlgoEntropy);
	   		
	   		DecisionPathList newDecPathList = new DecisionPathList();
	   		DecisionPathList.DecisionPath decPath = new DecisionPathList.DecisionPath(childStat.getTotalCount(), childStat.getStat(),
	   				childStat.getClassValPr());
	   		DecisionPathPredicate predicate = DecisionPathPredicate.createRootPredicate(ROOT_PATH);
	   		
	   		newDecPathList.addDecisionPath(decPath);
	   		
	   		//save new decision path list
	   		writeDecisioList(newDecPathList, "dtb.decision.file.path.out",  context.getConfiguration() );
	   	}
	   	
	   	/**
	   	 * @param context
	   	 * @throws IOException
	   	 */
	   	private void expandTree(Context context) throws IOException {
	   		Map<Double, Pair<String, Integer>> splits = new TreeMap<Double, Pair<String, Integer>>();
	   		
	   		//group by split
	   		infoContentBySplit();
	   		
	   		DecisionPathList newDecPathList = new DecisionPathList();
	   		boolean isAlgoEntropy = infoAlgorithm.equals("entropy");
	   		double parentStat = 0;
		    List< DecisionPathList.DecisionPathPredicate> predicates = null;
	   		
	   		//parent paths
	   		Map<String, Map<String, InfoContentStat>> splitInfoContent;
	   		for (String parentPath :  decPathsInfoContentBySplit.keySet()) {
	   			
	   			//parent decision path in existing tree
	   			DecisionPathList.DecisionPath parentDecPath = findParentDecisionPath(parentPath);
	   			if (null  == parentDecPath) {
	   				throw new IllegalStateException("parent decision path not found: "  + parentPath);
	   			}
	   			parentStat = parentDecPath.getInfoContent();
	   			 
	   			//splits
	   			double minInfoContent = 1000000;
   				String selectedSplit = null;
   				int selectedSplitAttr = -1;
   				splits.clear();
	   			splitInfoContent = decPathsInfoContentBySplit.get(parentPath);
	   			for (String splitId :  splitInfoContent.keySet()) {
	   				Map<String, InfoContentStat> predInfoContent = splitInfoContent.get(splitId);
	   				if (debugOn) {
	   					System.out.println("split: " + splitId);
	   				}
	   				
	   				//predicates
	   				double weightedInfoContent = 0;
	   				int totalCount = 0;
	   				int attr = -1;
	   				for (String predicate : predInfoContent.keySet()) {
	   					if (debugOn) {
		   					System.out.println("predicate: " + predicate);
	   					}
	   					
	   					attr = Integer.parseInt(predicate.split("\\s+")[0]);
	   					InfoContentStat stat = predInfoContent.get(predicate);
	   					weightedInfoContent += stat.processStat(isAlgoEntropy) * stat.getTotalCount();
	   					totalCount += stat.getTotalCount();
	   				}
	   				//average info content across splits
	   				double  avInfoContent = weightedInfoContent / totalCount;
	   				if (spltSelStrategy.equals(SPLIT_SEL_BEST)) {
		   				//pick split with  minimum info content
		   				if (avInfoContent  < minInfoContent) {
		   					minInfoContent = avInfoContent;
		   					selectedSplit = splitId;
		   					selectedSplitAttr = attr;
		   					if (debugOn) {
		   						System.out.println("selectedSplit: " +  selectedSplit +  "  selectedSplitAttr:  "  +  selectedSplitAttr  + 
		   								"  minInfoContent: " + minInfoContent);
		   					}
		   				}
	   				} else if (spltSelStrategy.equals(SPLIT_SEL_RANDOM_TOP)) {
	   					splits.put(avInfoContent, new Pair<String, Integer>(splitId, attr));
	   				} else {
	   					throw new IllegalStateException("ivalid split slection strategy");
	   				}
	   			}
	   			
	   			//select randomly from top k splits
	   			if (spltSelStrategy.equals(SPLIT_SEL_RANDOM_TOP)) {
	   				Pair<String, Integer> split =  selectRandomSplitFromTop(splits);
	   				selectedSplit = split.getLeft();
	   				selectedSplitAttr = split.getRight();
	   			}
	   			
	   			//expand based on selected split
   				Map<String, InfoContentStat> predInfoContent = splitInfoContent.get(selectedSplit);
   				if (debugOn) {
   					System.out.println("selected split: " + selectedSplit  +  " selected attribute: " +selectedSplitAttr );
   				}
   				
	   			//parent predicates
	   			List<DecisionPathList.DecisionPathPredicate> parentPredicates = 
	   					DecisionPathList.DecisionPathPredicate.createPredicates(parentPath, schema);
	   			 
	   			//generate new path based on predicates of selected split 
				FeatureField field = schema.findFieldByOrdinal(selectedSplitAttr);
   				for (String predicateStr : predInfoContent.keySet()) {
   					if (debugOn) {
   						System.out.println("predicate in selected split: " + predicateStr );
   					}
   					DecisionPathList.DecisionPathPredicate predicate = null;
   					if (field.isInteger()) {
   						predicate = DecisionPathList.DecisionPathPredicate.createIntPredicate(predicateStr);
   					} else if (field.isDouble()) {
   						predicate = DecisionPathList.DecisionPathPredicate.createDoublePredicate(predicateStr);
   					} else if (field.isCategorical()) {
   						predicate = DecisionPathList.DecisionPathPredicate.createCategoricalPredicate(predicateStr);
   					} 
   					
 					//append new predicate to parent predicate list
   				    predicates = new ArrayList< DecisionPathList.DecisionPathPredicate>();
					predicates.addAll(parentPredicates);
					predicates.add(predicate);
   				
  					//create new decision path
					InfoContentStat stat = predInfoContent.get(predicateStr);
   					boolean toBeStopped = pathStoppingStrategy.shouldStop(stat, parentStat, parentPredicates.size() + 1);
   					DecisionPathList.DecisionPath decPath = new DecisionPathList.DecisionPath(predicates, stat.getTotalCount(),
   							stat.getStat(),  toBeStopped, stat.getClassValPr());
   					newDecPathList.addDecisionPath(decPath);
   					
  				}
	   		}
	   		
	   		//save new decision path list
	   		writeDecisioList(newDecPathList, "dtb.decision.file.path.out",  context.getConfiguration() );
	   	}
	   	
	   	/**
	   	 * @param splits
	   	 * @return
	   	 */
	   	private Pair<String, Integer> selectRandomSplitFromTop(Map<Double, Pair<String, Integer>> splits) {
	   		List<Pair<String, Integer>> topSplits = new ArrayList<Pair<String, Integer>>();
			int i = 0;
			for (Double inforContent : splits.keySet()) {
				topSplits.add(splits.get(inforContent));
				if (++i == topSplitCount)
					break;
			}
			return BasicUtils.selectRandom(topSplits);
		}
	   	
	   	/**
	   	 * 
	   	 */
	   	private void infoContentBySplit() {
	   		decPathsInfoContentBySplit.clear();
	   		
	   		//parent paths
	   		for (String parentPath :  decPaths.keySet() ) {		
	   			//strip off slit IDs
	   			String filtParentPath = stripSplitIds(parentPath);
	   			
	   			Map<String, Map<String, InfoContentStat>> splitInfoContent = decPathsInfoContentBySplit.get(filtParentPath);
	   			if (null == splitInfoContent) {
	   				splitInfoContent = new HashMap<String, Map<String, InfoContentStat>>();
	   				decPathsInfoContentBySplit.put(filtParentPath, splitInfoContent);
	   			}
	   					
	   			//child paths
	   			Map<String, InfoContentStat> childStats = decPaths.get(parentPath);
	   			for (String pred :  childStats.keySet()) {
	   				String[] items = BasicUtils.splitOnFirstOccurence(pred, SPLIT_DELIM , true);
	   				String splitId = items[0];
	   				String predicate = items[1];
	   				if (debugOn) {
	   					System.out.println("parentPath: " + parentPath +   " splitId: " + splitId + " predicate: " + predicate);
	   				}
	   				Map<String, InfoContentStat> predInfoContent = splitInfoContent.get(splitId);
	   				if (null == predInfoContent) {
	   					predInfoContent = new HashMap<String, InfoContentStat>();
	   					splitInfoContent.put(splitId, predInfoContent);
	   				}
	   				predInfoContent.put(predicate, childStats.get(pred) );
	   				
	   			}
	   		}
	   	}
	   	
	   	/**
	   	 * @param decPath
	   	 * @return
	   	 */
	   	private  String stripSplitIds(String decPath) {
	   		String filtDecPath = null;
	   		String[] predicates = decPath.split(decPathDelim);
	   		String[] filtPredicates = new String[predicates.length];
	   		for (int i = 0; i < predicates.length; ++i )  {
	   			if (predicates[i].equals(ROOT_PATH)) {
	   				filtPredicates[i] = predicates[i];
	   			} else {
	   				filtPredicates[i] = BasicUtils.splitOnFirstOccurence(predicates[i], SPLIT_DELIM , true)[1];
	   			}
	   		}
	   		filtDecPath = BasicUtils.join(filtPredicates, decPathDelim);
	   		return filtDecPath;
	   	}
	   	
	   	/**
	   	 * finds decision path from current tree
	   	 * @param parentPath
	   	 * @return
	   	 */
	   	private DecisionPathList.DecisionPath findParentDecisionPath(String parentPath) {
	   		DecisionPathList.DecisionPath decPath =  null ;
	   		if (null != decPathList) {
	   			if (parentPath.equals(ROOT_PATH)) {
	   				decPath = decPathList.findDecisionPath(ROOT_PATH);
	   			} else {
	   		   		String[] parentPathItems = parentPath.split(decPathDelim);
	   		   		decPath = decPathList.findDecisionPath(parentPathItems);
	   			}
	   		}
	   		return decPath;
	   	}
	   	
	   	/**
	   	 * @param newDecPathList
	   	 * @param outFilePathParam
	   	 * @param conf
	   	 * @throws IOException
	   	 */
	   	private void writeDecisioList(DecisionPathList newDecPathList, String outFilePathParam, Configuration conf ) 
	   			throws IOException {
	   		//total population for support calculation
	   		if (totalPopulation > 0) {
	   			newDecPathList.withTotalPopulation(totalPopulation);
	   		}
	   		
	   		//save it
	   		FSDataOutputStream ouStrm = FileSystem.get(conf).create(new Path(conf.get(outFilePathParam)));	
	   		ObjectMapper mapper = new ObjectMapper();
	   		mapper.writeValue(ouStrm, newDecPathList);
	   		ouStrm.flush();
	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<Text> values, Context context)
        		throws IOException, InterruptedException {
        	int keySize = key.getSize();
        	key.setDelim(decPathDelim);
        	decPath = key.toString();
        	
        	if (keySize > 1) {
        		//tree exists
        		parentDecPath =  key.toString(0, keySize-1);
        		childPath = key.getString(keySize-1);
        	} else {
        		//tree does not exist
        		parentDecPath = key.getString(0);
        		childPath = CHILD_PATH;
        	}
        	
        	//all child class stats
        	Map<String, InfoContentStat> candidateChildrenPath =  decPaths.get(parentDecPath);
        	if (null == candidateChildrenPath) {
        		candidateChildrenPath = new HashMap<String, InfoContentStat>();
        		decPaths.put(parentDecPath,  candidateChildrenPath);
        	}
        	
        	//class stats
        	InfoContentStat classStats = candidateChildrenPath.get(childPath);
        	if (null == classStats) {
        		classStats = new InfoContentStat();
        		candidateChildrenPath.put(childPath, classStats);
        	}
        	
        	
        	for (Text value : values) {
        		classAttrValue = value.toString().split(fieldDelim)[classAttrOrdinal];
        		classStats.incrClassValCount(classAttrValue);
            	outVal.set(decPath + fieldDelim + value.toString());
            	context.write(NullWritable.get(), outVal);
        	}
        }
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new DecisionTreeBuilder(), args);
        System.exit(exitCode);
	}
	
}
