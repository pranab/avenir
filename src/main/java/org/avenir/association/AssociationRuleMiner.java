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

package org.avenir.association;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
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
import org.chombo.util.SecondarySort;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Association rule mining from frequent item sets
 * @author pranab
 *
 */
public class AssociationRuleMiner   extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(AssociationRuleMiner.class);

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Association rule mining from frequent item sets";
        job.setJobName(jobName);
        
        job.setJarByClass(AssociationRuleMiner.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(AssociationRuleMiner.RuleMinerMapper.class);
        job.setReducerClass(AssociationRuleMiner.RuleMinerReducer.class);
        
        job.setGroupingComparatorClass(SecondarySort.TuplePairGroupComprator.class);
        job.setPartitionerClass(SecondarySort.TuplePairPartitioner.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        
        int numReducer = job.getConfiguration().getInt("arm.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class RuleMinerMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private String fieldDelimRegex;
		private String[] items;
        private Tuple outKey = new Tuple();
		private Tuple outVal  = new Tuple();
        private int maxAntecedentSize;
        private List<String> transItems = new ArrayList<String>();;
        private double support;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	maxAntecedentSize = conf.getInt("arm.max.ante.size", 3);
       }

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	
       		transItems.clear();;
       		for (int i = 0; i < items.length-1; ++i) {
       			transItems.add(items[i]);
        	}
       		
       		//emit support
       		outKey.initialize();
       		outVal.initialize();
       		outKey.add(transItems);
       		outKey.add(Utility.ZERO);
       		
       		support = Double.parseDouble(items[items.length-1]);
       		outVal.add(support);
       		context.write(outKey, outVal);
       		
       		//generate  antecedent and consequent
       		if (transItems.size() > 1) {
       			//generate sub lists
       			List<List<String>>  subLists = Utility.generateSublists(transItems,  maxAntecedentSize);
       			for (List<String> subList : subLists) {
       	       		outKey.initialize();
       	       		outVal.initialize();
       	       		
       	       		//key ; antecedent 
       	       		outKey.add(subList);
       	       		outKey.add(Utility.ONE);
       	       		
       	       		//value: consequent and support for whole item set
       	       		List<String> diff = Utility.listDifference(transItems, subList);
       				outVal.add(diff);
       				outVal.add(support);
       	       		context.write(outKey, outVal);
       			}
       		}
        }       
 	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class RuleMinerReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private String fieldDelim;
		private Text outVal  = new Text();
		private double confidenceThreshold;
        private double confidence;
        private double totalSupport;
        private double anteSupport;

	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	protected void setup(Context context) 
	   			throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelim = conf.get("field.delim.out", ",");
            confidenceThreshold = Utility.assertDoubleConfigParam(conf, "arm.conf.threshold", "missing confidence threshold");
 	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<Tuple> values, Context context)
        		throws IOException, InterruptedException {
        	for (Tuple value : values) {
        		if (key.getLastAsInt() == 0) {
        			//antecedent support
        			anteSupport = value.getDouble(0);
        		} else {
        			//consequents
        			totalSupport = value.getLastAsDouble();
        			confidence = totalSupport / anteSupport;
        			if (confidence > confidenceThreshold) {
        				outVal.set(key.toString(0, key.getSize()-1)  + " -> " + value.toString(0, value.getSize()-1));
        				context.write(NullWritable.get(),outVal);
        			}
        		}
        	}
        }
	}	

	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new AssociationRuleMiner(), args);
        System.exit(exitCode);
	}
}
