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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.util.Utility;

/**
 * Does under sampling of majority class and makes class distribution balanced. Caches initial
 * set of records so we have a distribution to bootstrap from. Keeps track of counts for each class 
 * Dynamically under samples majority class
 * @author pranab
 *
 */
public class UnderSamplingBalancer extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Under samling class balancer ";
        job.setJobName(jobName);
        
        job.setJarByClass(UnderSamplingBalancer.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(UnderSamplingBalancer.SamplingMapper.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(0);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class SamplingMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
        private String fieldDelimRegex;
		private Map<String, Integer> classCounter = new HashMap<String, Integer>();
		private int classAttrOrd;
		private String[] items;
		private String classAttrValue;
		private Integer count;
		private List<Text> batch = new ArrayList<Text>();
		private int distrBatchSize;
		private int rowCount;
		private int minCount;
		
	    protected void setup(Context context) throws IOException, InterruptedException {
	    	Configuration conf = context.getConfiguration();
            fieldDelimRegex = context.getConfiguration().get("field.delim.regex", ",");
	    	classAttrOrd = conf.getInt("usb.class.attr.ord", -1);
	    	distrBatchSize = conf.getInt("usb.distr.batch.size", 500);
	    }	
	    
	    @Override
	    protected void map(LongWritable key, Text value, Context context)
	    	throws IOException, InterruptedException {
	    	++rowCount;
	    	
	    	//update distribution
	    	items  =  value.toString().split(fieldDelimRegex);
	    	classAttrValue = items[classAttrOrd];
	    	count = classCounter.get(classAttrValue);
	    	if (null == count) {
        	   count = 1;
	    	} else {
        	   ++count;
	    	}
	    	classCounter.put(classAttrValue, count);
	    	
	    	if (rowCount < distrBatchSize) {
	    		//add to batch and don't emit
	     	   	batch.add(new Text(value));
	    	} else if (rowCount == distrBatchSize) {
	    		//we have some stats, emit everything in batch
		    	minCount = getMinClassCount();
		    	int currentCount = count;
	    		for (Text row : batch) {
	    	    	items  =  row.toString().split(fieldDelimRegex);
	    	    	classAttrValue = items[classAttrOrd];
	    	    	count = classCounter.get(classAttrValue);
	    	    	emit(value, context);
	    		}
	    		batch.clear();
	    		
	    		//emit current
	    		count = currentCount;
    	    	emit(value, context);
	    	} else {
	    		//emit current
	    	   minCount = getMinClassCount();
	    	   emit(value, context);
	    	}
	    }
	    
	    /**
	     * @return
	     */
	    private int getMinClassCount() {
	    	int minCount = Integer.MAX_VALUE;
	    	for (String clAttrVal : classCounter.keySet()) {
	    		if (classCounter.get(clAttrVal) < minCount) {
	    			minCount = classCounter.get(clAttrVal);
	    		}
	    	}
	    	return minCount;
	    }
	    
	    /**
	     * @param value
	     * @param context
	     * @throws IOException
	     * @throws InterruptedException
	     */
	    private void emit(Text value, Context context) throws IOException, InterruptedException {
    	   if (count > minCount) {
    		   //majority classes
    		   double threshold = (double)minCount / count;
    		   if (Math.random() < threshold) {
        		   context.write(NullWritable.get(), value);
    		   }
    	   } else {
    		   //minority class
    		   context.write(NullWritable.get(), value);
    	   }
	    	
	    }
	}
	
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new UnderSamplingBalancer(), args);
        System.exit(exitCode);
	}
	
}
