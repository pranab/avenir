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
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.explore.ClassPartitionGenerator;
import org.avenir.explore.ClassPartitionGenerator.AttributeSplitPartitioner;
import org.avenir.explore.ClassPartitionGenerator.PartitionGeneratorMapper;
import org.avenir.explore.ClassPartitionGenerator.PartitionGeneratorReducer;
import org.avenir.tree.SplitManager.AttributePredicate;
import org.avenir.util.AttributeSplitHandler;
import org.avenir.util.AttributeSplitStat;
import org.avenir.util.InfoContentStat;
import org.chombo.mr.FeatureField;
import org.chombo.mr.FeatureSchema;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

public class DecisionTreeBuilder   extends Configured implements Tool {

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
        private int maxCatAttrSplitGroups;
        private SplitManager splitManager;
        private String attrSelectStrategy;
        private int randomSplitSetSize;
        private String classVal;
        private String currenttDecPath;
        private String decPathDelim;
        private static final Logger LOG = Logger.getLogger(BuilderMapper.class);

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	maxCatAttrSplitGroups = conf.getInt("max.cat.attr.split.groups", 3);
        	
        	//schema
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            
            //split manager
            decPathDelim = conf.get("dec.path.delim", ";");
            splitManager = new SplitManager(conf, "dec.path.file.path",  decPathDelim , schema); 
            
            //attribute selection strategy
            attrSelectStrategy = conf.get("split.attribute.selection.strategy", "notUsedYet");
 
           	randomSplitSetSize = conf.getInt("random.split.set.size", 3);
            
            //class attribute
            classField = schema.findClassAttrField();
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            classVal = items[classField.getOrdinal()];
            
            //get split attributes
           getSplitAttributes();
            
            currenttDecPath = null;
            if (splitManager.isTreeAvailable()) {
            	currenttDecPath = items[0];
            }
            
            //all attributes
            for (int attr :  splitAttrs) {
            	FeatureField field = schema. findFieldByOrdinal(attr);
            	Object attrValue = null;
            	//all splits
            	List<List<AttributePredicate>> allSplitPredicates = null;
            	if (field.isInteger()) {
            		allSplitPredicates = splitManager.createIntAttrSplitPredicates(attr);
            		Integer iValue = Integer.parseInt(items[attr]);
            		attrValue = iValue;
            	} else if (field.isDouble()) {
            		allSplitPredicates = splitManager.createDoubleAttrSplitPredicates(attr);
            		Double dValue = Double.parseDouble(items[attr]);
            		attrValue = dValue;
            	} else if (field.isCategorical()) {
            		allSplitPredicates = splitManager.createCategoricalAttrSplitPredicates(attr);
            		attrValue = items[attr];
            	}
                
            	//evaluate split predicates
                for (List<AttributePredicate> predicates : allSplitPredicates) {
                	for (AttributePredicate predicate : predicates) {
                		if (predicate.evaluate(attrValue)) {
                			//data belongs to this split segment
                			outKey.initialize();
                			if (null == currenttDecPath) {
                				outKey.add(predicate.toString());
                			} else {
                				String[] curDecPathItems = items[0].split(decPathDelim);
                				for (String curDecPathItem : curDecPathItems) {
                    				outKey.add(curDecPathItem);
                				}
                 				outKey.add(predicate.toString());
                			}               			}
                			outVal.set(value.toString());
            				context.write(outKey, outVal);
                		}	
                	}
                }
 		}

        /**
         * @param attrSelectStrategy
         * @param conf
         */
        private void getSplitAttributes() {
            if (attrSelectStrategy.equals("all")) {
            	//all attributes
            	splitAttrs = splitManager.getAllAttributes();
            } else if (attrSelectStrategy.equals("notUsedYet")) {
            	//attributes that have not been used yet
            	splitAttrs = splitManager.getRemainingAttributes();
            } else if (attrSelectStrategy.equals("randomAll")) {
            	//randomly selected k attributes from all
            	splitManager.getRandomAllAttributes(randomSplitSetSize);
            } else if (attrSelectStrategy.equals("randomNotUsedYet")) {
            	//randomly selected k attributes from attributes not used yet
            	splitManager.getRandomRemainingAttributes(randomSplitSetSize);
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
            
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
        	fieldDelim = conf.get("field.delim.out", ",");

        	infoAlgorithm = conf.get("split.algorithm", "giniIndex");
        	outputSplitProb = conf.getBoolean("output.split.prob", false);
	   	}   

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<Text> values, Context context)
        		throws IOException, InterruptedException {
        	
        }
	   	
	}
}
