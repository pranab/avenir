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


package org.avenir.explore;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.util.AttributeSplitHandler;
import org.avenir.util.AttributeSplitStat;
import org.avenir.util.InfoContentStat;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * Info content driven partitioning of attributes
 * @author pranab
 *
 */
public class ClassPartitionGenerator extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Candidate split generator for attributes";
        job.setJobName(jobName);
        job.setJarByClass(ClassPartitionGenerator.class);
        Utility.setConfiguration(job.getConfiguration(), "avenir");

        String[] paths = getPaths(args, job);
        FileInputFormat.addInputPath(job, new Path(paths[0]));
        FileOutputFormat.setOutputPath(job, new Path(paths[1]));
        
        job.setMapperClass(ClassPartitionGenerator.PartitionGeneratorMapper.class);
        job.setReducerClass(ClassPartitionGenerator.PartitionGeneratorReducer.class);
        job.setCombinerClass(ClassPartitionGenerator.PartitionGeneratorCombiner.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setPartitionerClass(AttributeSplitPartitioner.class);
        
        job.setNumReduceTasks(job.getConfiguration().getInt("num.reducer", 1));

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	/**
	 * Uses user provided paths
	 * @param args
	 * @param job
	 * @return
	 */
	protected String[] getPaths(String[] args, Job job) {
		String[] paths = new String[2];
		paths[0] = args[0];
		paths[1] = args[1];
		return paths;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class PartitionGeneratorMapper extends Mapper<LongWritable, Text, Tuple, IntWritable> {
		private String fieldDelimRegex;
		private String[] items;
        private Tuple outKey = new Tuple();
		private IntWritable outVal  = new IntWritable(1);
        private FeatureSchema schema;
        private int[] splitAttrs;
        private AttributeSplitHandler splitHandler = new AttributeSplitHandler();
        private FeatureField classField;
        private boolean atRoot = false;
        private int maxCatAttrSplitGroups;
        private static final Logger LOG = Logger.getLogger(PartitionGeneratorMapper.class);

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	
        	maxCatAttrSplitGroups = conf.getInt("cpg.max.cat.attr.split.groups", 3);
        	
        	//schema
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "cpg.feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            
            //attribute selection strategy
            String attrSelectStrategy = conf.get("cpg.split.attribute.selection.strategy", "userSpecified");
            
            //get split attributes
            getSplitAttributes(attrSelectStrategy, conf);
            
            //generate all attribute splits
            if (!atRoot) {
	            createPartitions();
	            LOG.debug("created split partitions");
            }
            
            //class attribute
            classField = schema.findClassAttrField();
        }
        
        /**
         * @param attrSelectStrategy
         * @param conf
         */
        private void getSplitAttributes(String attrSelectStrategy, Configuration conf) {
            atRoot = conf.getBoolean("cpg.at.root", false);
            if (atRoot) {
            	LOG.debug("processing at root");
            } else if (attrSelectStrategy.equals("userSpecified")) {
            	//user specified attributes
	            String attrs = conf.get("cpg.split.attributes");
	            splitAttrs = Utility.intArrayFromString(attrs, ",");
            } else if (attrSelectStrategy.equals("all")) {
            	//all attributes
            	splitAttrs = schema.getFeatureFieldOrdinals();
            } else if (attrSelectStrategy.equals("notUsedYet")) {
            	//attributes that have not been used yet
            	int[] allSplitAttrs = schema.getFeatureFieldOrdinals();
            	int[] usedSplitAppributes = null; //TODO
            	splitAttrs = Utility.removeItems(allSplitAttrs, usedSplitAppributes);
            	
            } else if (attrSelectStrategy.equals("random")) {
            	//randomly selected k attributes
            	int randomSplitSetSize = conf.getInt("cpg.random.split.set.size", 3);
               	int[] allSplitAttrs = schema.getFeatureFieldOrdinals();
               	Set<Integer> splitSet = new  HashSet<Integer>();
               	while (splitSet.size() != randomSplitSetSize) {
               		int splitIndex = (int)(Math.random() * allSplitAttrs.length);
               		splitSet.add(allSplitAttrs[splitIndex]);
               	}
               	
               	splitAttrs = new int[randomSplitSetSize];
               	int i = 0;
               	for (int spAttr : splitSet) {
               		splitAttrs[i++] =  spAttr;
               	}
            } else {
            	throw new IllegalArgumentException("invalid splitting attribute selection strategy");
            }
        	
        }
        
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            String classVal = items[classField.getOrdinal()];
            
            //all attributes
            if (atRoot) {
				outKey.initialize();
				outKey.add(-1, "null", -1,classVal);
				context.write(outKey, outVal);
            } else {
            	//all attributes
	            for (int attrOrd : splitAttrs) {
	            	Integer attrOrdObj = attrOrd;
	        		FeatureField featFld = schema.findFieldByOrdinal(attrOrd);
	        		
        			String attrValue = items[attrOrd];
        			splitHandler.selectAttribute(attrOrd);
        			String splitKey = null;
        			
        			//all splits
        			while((splitKey = splitHandler.next()) != null) {
        				Integer segmentIndex = splitHandler.getSegmentIndex(attrValue);
        				outKey.initialize();
        				outKey.add(attrOrdObj, splitKey, segmentIndex,classVal);
        				context.write(outKey, outVal);
        				context.getCounter("Stats", "mapper output count").increment(1);
        			}
	            }
            }
        }
        
        /**
         * create partitions of  different sizes
         */
        private void createPartitions() {
        	for (int attrOrd : splitAttrs) {
        		FeatureField featFld = schema.findFieldByOrdinal(attrOrd);
        		if (featFld.isInteger()) {
        			//numerical
        			List<Integer[]> splitList = new ArrayList<Integer[]>();
        			Integer[] splits = null;
        			createNumPartitions(splits, featFld, splitList);
        			
        			//collect all splits
        			for (Integer[] thisSplit : splitList) {
        				splitHandler.addIntSplits(attrOrd, thisSplit);
        			}
        		} else if (featFld.isCategorical()) {
        			//categorical
        			int numGroups = featFld.getMaxSplit();
        			if (numGroups > maxCatAttrSplitGroups) {
        				throw new IllegalArgumentException(
        					"more than " +  maxCatAttrSplitGroups + " split groups not allwed for categorical attr");
        			}
        			
        			//try all group count from 2 to max
        			List<List<List<String>>> finalSplitList = new ArrayList<List<List<String>>>();
        			for (int gr = 2; gr <= numGroups; ++gr) {
        				LOG.debug("num of split sets:" + gr);
        				List<List<List<String>>> splitList = new ArrayList<List<List<String>>>();
        				createCatPartitions(splitList,  featFld.getCardinality(), 0, gr);
        				finalSplitList.addAll(splitList);
        			}
        			
        			//collect all splits
        			for (List<List<String>> splitSets : finalSplitList) {
        				splitHandler.addCategoricalSplits(attrOrd, splitSets);
        			}
        			
        		}
        	}
        }
        
        /**
         * Create all possible splits within the max number of splits allowed
         * @param splits previous split
         * @param featFld
         * @param newSplitList all possible splits
         */
        private void createNumPartitions(Integer[] splits, FeatureField featFld, 
        		List<Integer[]> newSplitList) {
    		int min = (int)(featFld.getMin() + 0.01);
    		int max = (int)(featFld.getMax() + 0.01);
    		int binWidth = featFld.getBucketWidth();
        	if (null == splits) {
        		//first time
        		for (int split = min + binWidth ; split < max; split += binWidth) {
        			Integer[] newSplits = new Integer[1];
        			newSplits[0] = split;
        			newSplitList.add(newSplits);
        			createNumPartitions(newSplits, featFld,newSplitList);
        		}
        	} else {
        		//create split based off last split that will contain one additinal split point
        		int len = splits.length;
        		if (len < featFld.getMaxSplit() -1) {
	        		for (int split = splits[len -1] + binWidth; split < max; split += binWidth) {
	        			Integer[] newSplits = new Integer[len + 1];
	        			int i = 0;
	        			for (; i < len; ++i) {
	        				newSplits[i] = splits[i];
	        			}
	        			newSplits[i] = split;
	        			newSplitList.add(newSplits);
	        			
	        			//recurse to generate additional splits
	        			createNumPartitions(newSplits, featFld,newSplitList);
	        		}
        		}
        	}
        }
        
        /**
         * @param splits
         * @param featFld
         * @param newSplitList
         */
        private void createCatPartitions(List<List<List<String>>>  splitList, List<String> cardinality,
        		int cardinalityIndex, int numGroups) {
        	LOG.debug("next round cardinalityIndex:" + cardinalityIndex);
    		//first time
        	if (0 == cardinalityIndex) {
    			//initial full splits
    			List<List<String>> fullSp = createInitialSplit(cardinality, numGroups);

    			//partial split shorter in length by one 
    			List<List<List<String>>> partialSpList = createPartialSplit(cardinality,numGroups-1, numGroups);
    			
    			//split list
    			splitList.add(fullSp);
    			splitList.addAll(partialSpList);
    			
    			//recurse
    			cardinalityIndex += numGroups;
    			createCatPartitions(splitList, cardinality,cardinalityIndex, numGroups);
    		//more elements to consume	
        	} else if (cardinalityIndex < cardinality.size()){
        		List<List<List<String>>>  newSplitList = new ArrayList<List<List<String>>>(); 
        		String newElement = cardinality.get(cardinalityIndex);
        		for (List<List<String>> sp : splitList) {
        			if (sp.size() == numGroups) {
        				//if full split, append new element to each group within split to create new splits
        				LOG.debug("creating new split from full split");
        				for (int i = 0; i < numGroups; ++i) {
            				List<List<String>> newSp = new ArrayList<List<String>>();
        					for (int j = 0; j < sp.size(); ++j) {
        						List<String> gr = cloneStringList(sp.get(j));
        						if (j == i) {
        							//add new element
        							gr.add(newElement);
        						}
        						newSp.add(gr);
        					}
        					newSplitList.add(newSp);
        				}
        			} else {
        				//if partial split, create new group with new element and add to split
        				LOG.debug("creating new split from partial split");
        				List<List<String>> newSp = new ArrayList<List<String>>();
    					for (int i = 0; i < sp.size(); ++i) {
    						List<String> gr = cloneStringList(sp.get(i));
    						newSp.add(gr);
    					}
    					List<String> newGr = new ArrayList<String>();
    					newGr.add(newElement);
    					newSp.add(newGr);
    					newSplitList.add(newSp);
        			}
        			LOG.debug("newSplitList:" + newSplitList);
        		}
        		
        		//generate partial splits
        		if (cardinalityIndex < cardinality.size() -1){        		
        			List<List<List<String>>> partialSpList = createPartialSplit(cardinality,cardinalityIndex, numGroups);
    				newSplitList.addAll(partialSpList);
        		}
        		
        		//replace old splits with new
				splitList.clear();
				splitList.addAll(newSplitList);
				
    			//recurse
				++cardinalityIndex;
    			createCatPartitions(splitList, cardinality,cardinalityIndex, numGroups);
        	}
        }	
    	
        /**
         * @param cardinality
         * @param numGroups
         * @return
         */
        private List<List<String>> createInitialSplit(List<String> cardinality, int numGroups) {
        	List<List<String>> newSp = new ArrayList<List<String>>();
    		for (int i = 0; i < numGroups; ++i) {
    			List<String> gr = new ArrayList<String>();
    			gr.add(cardinality.get(i));
    			newSp.add(gr);
    		}
    		LOG.debug("initial split:" + newSp);
        	return newSp;
        }
        
        /**
         * @param cardinality
         * @param cardinalityIndex
         * @param numGroups
         * @return
         */
        private List<List<List<String>>> createPartialSplit(List<String> cardinality,
        		int cardinalityIndex, int numGroups) {
			List<List<List<String>>> partialSplitList = new ArrayList<List<List<String>>>();
        	if (numGroups == 2) {
            	List<List<String>> newSp = new ArrayList<List<String>>();
        		List<String> gr = new ArrayList<String>();
        		for (int i = 0;i <= cardinalityIndex; ++i) {
        			gr.add(cardinality.get(i));
        		}
        		newSp.add(gr);
        		partialSplitList.add(newSp);
        	} else {
        		//create split list with splits shorter in length by 1
        		List<String> partialCardinality = new ArrayList<String>();
        		for (int i = 0; i <= cardinalityIndex; ++i) {
        			partialCardinality.add(cardinality.get(i));
        		}
    			createCatPartitions(partialSplitList,  partialCardinality, 0, numGroups-1);
        	}
        	
    		LOG.debug("partial split:" + partialSplitList);
        	return partialSplitList;
        }
        
        /**
         * @param curList
         * @return
         */
        private List<String> cloneStringList(List<String> curList) {
        	List<String> newList = new ArrayList<String>();
        	newList.addAll(curList);
        	return newList;
        }
	}

    
	/**
	 * @author pranab
	 *
	 */
	public static class PartitionGeneratorCombiner extends Reducer<Tuple, IntWritable, Tuple, IntWritable> {
		private int count;
		private IntWritable outVal = new IntWritable();
		
        protected void reduce(Tuple  key, Iterable<IntWritable> values, Context context)
        		throws IOException, InterruptedException {
        	count = 0;
        	for (IntWritable value : values) {
        		count += value.get();
        	}
        	outVal.set(count);
        	context.write(key, outVal);       	
        }		
	}	
	
	/**
	 * @author pranab
	 *
	 */
	public static class PartitionGeneratorReducer extends Reducer<Tuple, IntWritable, NullWritable, Text> {
 		private FeatureSchema schema;
		private String fieldDelim;
		private Text outVal  = new Text();
		private Map<Integer, AttributeSplitStat> splitStats = new HashMap<Integer, AttributeSplitStat>();
		private InfoContentStat rootInfoStat;
		private int count;
		private int[] attrOrdinals;
		private String  infoAlgorithm;
        private boolean atRoot = false;
        private boolean outputSplitProb;
        private double parentInfo;
        private static final Logger LOG = Logger.getLogger(PartitionGeneratorReducer.class);
        
	   	@Override
	   	protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            	AttributeSplitStat.enableLog();
            }
            
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "cpg.feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
        	fieldDelim = conf.get("field.delim.out", ",");

        	infoAlgorithm = conf.get("cpg.split.algorithm", "giniIndex");
            String attrs = conf.get("cpg.split.attributes");
            if (null != attrs) {
            	//attribute level
            	attrOrdinals = Utility.intArrayFromString(attrs, ",");
            	for (int attrOrdinal : attrOrdinals) {
            		splitStats.put(attrOrdinal, new AttributeSplitStat(attrOrdinal, infoAlgorithm));
            	}
            } else {
            	//data set root level
            	atRoot = true;
            	rootInfoStat = new InfoContentStat();
            }
        	outputSplitProb = conf.getBoolean("cpg.output.split.prob", false);
        	parentInfo = Double.parseDouble(conf.get("cpg.parent.info"));
	   	}   
	   	
	   	@Override
	   	protected void cleanup(Context context)  throws IOException, InterruptedException {
	   		//get stats and emit
	   		if (atRoot) {
	   			double stat = rootInfoStat.processStat(infoAlgorithm.equals("entropy"));
   				outVal.set("" + stat);
   				context.write(NullWritable.get(),outVal);
	   		}  else {
	   			double stat = 0;
	   			double gain = 0;
	   			double gainRatio = 0;
		   		for (int attrOrdinal : attrOrdinals) {
		   			AttributeSplitStat splitStat = splitStats.get(attrOrdinal);
		   			Map<String, Double> stats = splitStat.processStat(infoAlgorithm);
		   			for (String key : stats.keySet()) {
		   				StringBuilder stBld = new StringBuilder();
		   				stat = stats.get(key);
		   				
		   				if (infoAlgorithm.equals(AttributeSplitStat.ALG_ENTROPY) || 
		   						infoAlgorithm.equals(AttributeSplitStat.ALG_GINI_INDEX)) {
			   				gain = parentInfo - stat;
			   				gainRatio = gain / splitStat.getInfoContent(key);
			   				LOG.debug("attrOrdinal:" + attrOrdinal + " splitKey:" + key + " stat:" + stat +
			   						" gain:"  + gain + " gainRatio:" + gainRatio);
			   				
			   				stBld.append(attrOrdinal).append(fieldDelim).append(key).append(fieldDelim).append(gainRatio);
			   				if (outputSplitProb) {
				   				Map<Integer, Map<String, Double>> classValPr = splitStat.getClassProbab(key);
			   					stBld.append(fieldDelim).append(serializeClassProbab(classValPr));
			   				}
		   				} else {
			   				stBld.append(attrOrdinal).append(fieldDelim).append(key).append(fieldDelim).append(stat);
		   				}
		   				
		   				outVal.set(stBld.toString());
		   				context.write(NullWritable.get(),outVal);
		   			}
		   		}
	   		}
			super.cleanup(context);
	   	}
	   	
	   	private String serializeClassProbab(Map<Integer, Map<String, Double>> classValPr) {
	   		StringBuilder stBld = new StringBuilder();
	   		for (Integer splitSegment : classValPr.keySet()) {
	   			Map<String, Double> classPr = classValPr.get(splitSegment);
	   			for (String classVal : classPr.keySet()) {
	   				stBld.append(splitSegment).append(fieldDelim).append(classVal).append(fieldDelim);
	   				stBld.append(classPr.get(classVal)).append(fieldDelim);
	   			}
	   		}
	   		
	   		return stBld.substring(0, stBld.length() -1);
	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<IntWritable> values, Context context)
        		throws IOException, InterruptedException {
			context.getCounter("Stats", "reducer input count").increment(1);
        	int attrOrdinal = key.getInt(0);
        	String splitKey = key.getString(1);
        	int segmentIndex = key.getInt(2);
        	String classVal = key.getString(3);
        	
        	count = 0;
        	for (IntWritable value : values) {
        		count += value.get();
        	}
        	if (atRoot) {
        		rootInfoStat.countClassVal(classVal, count);
        	} else {
	        	AttributeSplitStat splitStat = splitStats.get(attrOrdinal);
	        	//LOG.debug("In reducer attrOrdinal:" + attrOrdinal + " splitKey:" + splitKey + 
	        	//		" segmentIndex:" + segmentIndex + " classVal:" + classVal);
	        	//update count
	        	splitStat.countClassVal(splitKey, segmentIndex, classVal, count);
        	}
        }	   	
        
	}	
	
    /**
     * @author pranab
     *
     */
    public static class AttributeSplitPartitioner extends Partitioner<Tuple, IntWritable> {
	     @Override
	     public int getPartition(Tuple key,  IntWritable value, int numPartitions) {
	    	 //consider only first 2 components of the key
		     return key.hashCodePartial(2) % numPartitions;
	     }
   }
    
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new ClassPartitionGenerator(), args);
        System.exit(exitCode);
	}    

}
