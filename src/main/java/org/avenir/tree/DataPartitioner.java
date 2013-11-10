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
import java.util.Arrays;
import java.util.List;

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
import org.avenir.explore.ClassPartitionGenerator.AttributeSplitPartitioner;
import org.avenir.util.AttributeSplitHandler;
import org.chombo.mr.FeatureField;
import org.chombo.mr.FeatureSchema;
import org.chombo.util.SecondarySort;
import org.chombo.util.TextInt;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * Partitions data based on a split selected among some candidate splits
 * generated from the parent node and corresponding data
 * @author pranab
 *
 */
public class DataPartitioner extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Candidate split generator for attributes";
        job.setJobName(jobName);
        
        job.setJarByClass(DataPartitioner.class);

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(DataPartitioner.PartitionerMapper.class);
        job.setReducerClass(DataPartitioner.PartitionerReducer.class);

        //find best split
        String inPath = args[0];
        Split split = findBestSplitKey(job, inPath);
        String outPath = inPath + "/" + "split=" + split.getSplitKey();
        		
        FileInputFormat.addInputPath(job, new Path(inPath));
        FileOutputFormat.setOutputPath(job, new Path(outPath));
        
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        
        job.setPartitionerClass(SecondarySort.RawIntKeyTextPartitioner.class);
        
        job.setNumReduceTasks(split.getSegmentCount());

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	/**
	 * Finds best split according to chosen strategy
	 * @param job
	 * @param iplutPath
	 * @return
	 * @throws IOException
	 */
	private Split findBestSplitKey(Job job, String iplutPath) throws IOException {
		String splitKey = null;
		Configuration conf =  job.getConfiguration();
		String splitSelectionStrategy = conf.get("split.selection.strategy", "best");
		
		String candidateSplitsPath = Utility.getSiblingPath(iplutPath, "candidateSplits/part-r-00000");
		conf.set("candidate.splits.path", candidateSplitsPath);
		List<String> lines = Utility.getFileLines(conf, "candidate.splits.path");
		
		//create split objects and sort
		Split[] splits = new Split[lines.size()];
		int i = 0;
		for (String line : lines) {
			splits[i++] = new Split(line);
		}
		
		//sort splits
		Arrays.sort(splits);

		//find split
		int splitIndex = 0;
		if (splitSelectionStrategy.equals("best")) {
		} else if (splitSelectionStrategy.equals("randomFromTop")) {
			int numSplits = conf.getInt("num.top.splits", 5);
			splitIndex = (int)(Math.random() * numSplits);
		}
		Split split = splits[splitIndex];
		splitKey = split.getSplitKey();
		
		//set asplit attribute ordinal and split key
		conf.setInt("split.attribute", split.getAttributeOrdinal());
		conf.set("split.key", splitKey);
		
        return split;
	}
	
	/**
	 * Sortable split
	 * @author pranab
	 *
	 */
	private static class Split implements  Comparable<Split> {
		private String line;
		private String[] items;
		
		public Split(String line) {
			this.line = line;
			items = line.split(",");
		}
		
		@Override
		public int compareTo(Split that) {
			double thisVal = Double.parseDouble(items[2]);
			double thatVal = Double.parseDouble(that.items[2]);
			
			//descending order
			return thisVal > thatVal ? -1 : (thisVal < thatVal ? 1 : 0);
		}
		
		private String getSplitKey() {
			return items[1];
		}
		
		private int getAttributeOrdinal() {
			return Integer.parseInt(items[0]);
		}
		
		public int getSegmentCount() {
			String[] segments = items[1].split(":");
			return segments.length;
		}
	}
	
    /**
     * @author pranab
     *
     */
    public static class RawIntKeyTextPartitioner extends Partitioner<IntWritable, Text> {
	     @Override
	     public int getPartition(IntWritable key, Text value, int numPartitions) {
	    	 //consider only base part of  key
		     return key.get();
	     }
    }
	
	
	/**
	 * @author pranab
	 *
	 */
	public static class PartitionerMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
		private String fieldDelimRegex;
		private String[] items;
		private IntWritable outKey = new IntWritable();
        private Text outVal = new Text();
        private FeatureSchema schema;
		private int splitAttrOrd;
		private FeatureField featureField;
		AttributeSplitHandler.CategoricalSplit catSplit;
		private int splitSegment;
		private String catAttrVal;
		
        private static final Logger LOG = Logger.getLogger(PartitionerMapper.class);
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	
        	splitAttrOrd = conf.getInt("split.attribute", -1);
        	if (splitAttrOrd == 1) {
        		throw new IOException("split attribute not found");
        	}
        	String splitKey = conf.get("split.key");
        	
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            featureField = schema.findFieldByOrdinal(splitAttrOrd);
            if (featureField.isInteger()) {
            	
            } else if (featureField.isCategorical()) {
            	catSplit = new AttributeSplitHandler.CategoricalSplit(splitKey);
            	catSplit.fromString();
            }
        	
            
            //new MultipleOutputs(conf);
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            if (featureField.isInteger()) {
            	
            } else if (featureField.isCategorical()) {
            	catAttrVal = items[splitAttrOrd];
            	splitSegment = catSplit.getSegmentIndex(catAttrVal);
            	outKey.set(splitSegment);
            }
			context.write(outKey,value);
        }        
	}
	
	
	/**
	 * @author pranab
	 *
	 */
	public static class PartitionerReducer extends Reducer<IntWritable, Text, NullWritable, Text> {
		
        protected void reduce(IntWritable  key, Iterable<Text> values, Context context)
        		throws IOException, InterruptedException {
        	for (Text value : values) {
        		context.write(NullWritable.get(), value);
        	}
        }
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new DataPartitioner(), args);
        System.exit(exitCode);
	}    
	
}
