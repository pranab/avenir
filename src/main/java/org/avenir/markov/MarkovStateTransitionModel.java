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

package org.avenir.markov;

import java.io.IOException;

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
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.util.StateTransitionProbability;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Markov state transition probability matrix
 * @author pranab
 *
 */
public class MarkovStateTransitionModel extends Configured implements Tool {
	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Markov tate transition model";
        job.setJobName(jobName);
        
        job.setJarByClass(MarkovStateTransitionModel.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(MarkovStateTransitionModel.StateTransitionMapper.class);
        job.setReducerClass(MarkovStateTransitionModel.StateTransitionReducer.class);
        job.setCombinerClass(MarkovStateTransitionModel.StateTransitionCombiner.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setNumReduceTasks(job.getConfiguration().getInt("num.reducer", 1));

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class StateTransitionMapper extends Mapper<LongWritable, Text, Tuple, IntWritable> {
		private String fieldDelimRegex;
		private String[] items;
		private int skipFieldCount;
        private Tuple outKey = new Tuple();
		private IntWritable outVal  = new IntWritable(1);

        private static final Logger LOG = Logger.getLogger(StateTransitionMapper.class);

        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
            skipFieldCount = conf.getInt("skip.field.count", 0);
        }
        
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	if (items.length >= (skipFieldCount + 2)) {
	        	for (int i = skipFieldCount + 1; i < items.length; ++i) {
	        		outKey.initialize();
	        		outKey.add(items[i-1], items[i]);
	        		context.write(outKey, outVal);
	        	}
        	}
        }        
        
	}	
	
	public static class StateTransitionCombiner extends Reducer<Tuple, IntWritable, Tuple, IntWritable> {
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
	public static class StateTransitionReducer extends Reducer<Tuple, IntWritable, NullWritable, Text> {
		private String fieldDelim;
		private Text outVal  = new Text();
		private String[] states;
		private StateTransitionProbability transProb;
		private int count;
		
		private static final Logger LOG = Logger.getLogger(StateTransitionMapper.class);
		
	   	protected void setup(Context context) 
	   			throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelim = conf.get("field.delim.out", ",");
        	states = conf.get("model.states").split(",");
        	transProb = new StateTransitionProbability(states, states);
        	int transProbScale = conf.getInt("trans.prob.scale", 1000);
        	transProb.setScale(transProbScale);
	   	}
	   	
	   	protected void cleanup(Context context)  
	   			throws IOException, InterruptedException {
	   		//all states
        	Configuration conf = context.getConfiguration();
	   		outVal.set(conf.get("model.states"));
   			context.write(NullWritable.get(),outVal);

   			//state transitions
   			transProb.normalizeRows();
	   		for (int i = 0; i < states.length; ++i) {
	   			String val = transProb.serializeRow(i);
	   			outVal.set(val);
	   			context.write(NullWritable.get(),outVal);
	   		}
	   	}
	   	
        protected void reduce(Tuple  key, Iterable<IntWritable> values, Context context)
        		throws IOException, InterruptedException {
        	count = 0;
        	for (IntWritable value : values) {
        		count += value.get();
        	}
        	String fromSt = key.getString(0);
        	String toSt = key.getString(1);
        	transProb.add(fromSt, toSt, count);
        }	   	
	}
	
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new MarkovStateTransitionModel(), args);
        System.exit(exitCode);
	}
	
	
}
