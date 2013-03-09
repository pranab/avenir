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
import java.util.List;
import java.util.Map;

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
import org.chombo.mr.FeatureField;
import org.chombo.mr.FeatureSchema;
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
        String jobName = "Partition generator for attributes";
        job.setJobName(jobName);
        
        job.setJarByClass(CramerCorrelation.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(ClassPartitionGenerator.PartitionGeneratorMapper.class);
        job.setReducerClass(ClassPartitionGenerator.PartitionGeneratorReducer.class);
        
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
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            
            //generate all attribute splits
            splitAttrs = Utility.intArrayFromString(conf.get("split.attributes"), ",");
            createPartitions();
            
            //class attribute
            classField = schema.findClassAttrField();
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            String classVal = items[classField.getOrdinal()];
            
            //all attributes
            for (int attrOrd : splitAttrs) {
            	Integer attrOrdObj = attrOrd;
        		FeatureField featFld = schema.findFieldByOrdinal(attrOrd);
        		if (featFld.isInteger()) {
        			String attrValue = items[attrOrd];
        			splitHandler.selectAttribute(attrOrd);
        			String splitKey = null;
        			
        			//all splits
        			while((splitKey = splitHandler.next()) != null) {
        				Integer segmentIndex = splitHandler.getSegmentIndex(attrValue);
        				outKey.initialize();
        				outKey.add(attrOrdObj, splitKey, segmentIndex,classVal);
        				context.write(outKey, outVal);
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
        			List<Integer[]> splitList = new ArrayList<Integer[]>();
        			Integer[] splits = null;
        			createNumPartitions(splits, featFld, splitList);
        			for (Integer[] thisSplit : splitList) {
        				splitHandler.addIntSplits(attrOrd, thisSplit);
        			}
        		} else if (featFld.isCategorical()) {
        			//TODO
        		}
        	}
        }
        
        /**
         * @param splits
         * @param featFld
         * @param newSplitList
         */
        private void createNumPartitions(Integer[] splits, FeatureField featFld, 
        		List<Integer[]> newSplitList) {
    		int min = featFld.getMin();
    		int max = featFld.getMax();
    		int binWidth = featFld.getBucketWidth();
        	if (null == splits) {
        		for (int split = min + binWidth ; split < max; split += binWidth) {
        			Integer[] newSplits = new Integer[1];
        			newSplits[0] = split;
        			newSplitList.add(newSplits);
        			createNumPartitions(newSplits, featFld,newSplitList);
        		}
        	} else {
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
	        			createNumPartitions(newSplits, featFld,newSplitList);
	        		}
        		}
        	}
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
		private int count;
		private int[] attrOrdinals;
		private String  infoAlgorithm;
        private static final Logger LOG = Logger.getLogger(PartitionGeneratorReducer.class);
        
	   	@Override
	   	protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
        	fieldDelim = conf.get("field.delim.out", ",");
        	
        	attrOrdinals = Utility.intArrayFromString(conf.get("split.attributes"), ",");
        	for (int attrOrdinal : attrOrdinals) {
        		splitStats.put(attrOrdinal, new AttributeSplitStat(attrOrdinal));
        	}
        	infoAlgorithm = conf.get("info.content.algorithm", "giniIndex");
	   	}   
	   	
	   	@Override
	   	protected void cleanup(Context context)  throws IOException, InterruptedException {
	   		//get stats and emit
	   		for (int attrOrdinal : attrOrdinals) {
	   			AttributeSplitStat splitStat = splitStats.get(attrOrdinal);
	   			Map<String, Double> stats = splitStat.processStat(infoAlgorithm.equals("entropy"));
	   			for (String key : stats.keySet()) {
	   				double stat = stats.get(key);
	   				outVal.set("" + attrOrdinal + fieldDelim + key + fieldDelim + stat);
	   				context.write(NullWritable.get(),outVal);
	   			}
	   		}
			super.cleanup(context);
	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<IntWritable> values, Context context)
        		throws IOException, InterruptedException {
        	int attrOrdinal = key.getInt(0);
        	String splitKey = key.getString(1);
        	int segmentIndex = key.getInt(2);
        	String classVal = key.getString(3);
        	AttributeSplitStat splitStat = splitStats.get(attrOrdinal);
        	
        	count = 0;
        	for (IntWritable value : values) {
        		count += value.get();
        	}
        	
        	//update count
        	splitStat.countClassVal(splitKey, segmentIndex, classVal, count);
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
