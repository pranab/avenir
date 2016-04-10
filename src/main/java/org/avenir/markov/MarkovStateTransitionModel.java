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
import java.util.HashMap;
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
 * Markov state transition probability matrix. Can also generate separate matrix for each
 * class label
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

        int numReducer = job.getConfiguration().getInt("mst.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);

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
		private int classLabelFieldOrd;
		private String classLabel;

        private static final Logger LOG = Logger.getLogger(StateTransitionMapper.class);

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
            skipFieldCount = conf.getInt("mst.skip.field.count", 0);
            classLabelFieldOrd = conf.getInt("mst.class.label.field.ord", -1);
            if (classLabelFieldOrd >= 0) {
            	++skipFieldCount;
            }
            
        }
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	if (items.length >= (skipFieldCount + 2)) {
	        	for (int i = skipFieldCount + 1; i < items.length; ++i) {
	        		outKey.initialize();
	        		if (classLabelFieldOrd >= 0) {
	        			//class label based markov model
	        			classLabel = items[classLabelFieldOrd];
	        			outKey.add(classLabel,items[i-1], items[i]);
	        		} else {
	        			//global markov model
	        			outKey.add(items[i-1], items[i]);
	        		}
        			context.write(outKey, outVal);
	        	}
        	}
        }        
        
	}	
	
	/**
	 * @author pranab
	 *
	 */
	public static class StateTransitionCombiner extends Reducer<Tuple, IntWritable, Tuple, IntWritable> {
		private int count;
		private IntWritable outVal = new IntWritable();
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
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
		private Map<String, StateTransitionProbability> classBasedTransProb =
				new HashMap<String, StateTransitionProbability>();
		private int count;
		private int transProbScale;
		private boolean isClassBasedModel;;
		private String classLabel;
		private String fromSt;
		private String toSt;
		private boolean outputStates;
		
		private static final Logger LOG = Logger.getLogger(StateTransitionMapper.class);
		
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
        	states = conf.get("mst.model.states").split(",");
        	transProb = new StateTransitionProbability(states, states);
        	transProbScale = conf.getInt("mst.trans.prob.scale", 1000);
        	transProb.setScale(transProbScale);
        	isClassBasedModel = conf.getInt("mst.class.label.field.ord", -1) >= 0;
        	outputStates = conf.getBoolean("mst.output.states", true); 
	   	}
	   	
	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	protected void cleanup(Context context)  
	   			throws IOException, InterruptedException {
	   		//all states
        	Configuration conf = context.getConfiguration();
        	if (outputStates) {
        		outVal.set(conf.get("mst.model.states"));
        		context.write(NullWritable.get(),outVal);
        	}
        	
   			//state transitions
   			if (isClassBasedModel) {
   				//class based model
   				for (String classLabel : classBasedTransProb.keySet()) {
   					StateTransitionProbability clsTransProb = classBasedTransProb.get(classLabel);
   					outVal.set("classLabel:" + classLabel);
   		   			context.write(NullWritable.get(),outVal);
   	   				outputPorbMatrix(clsTransProb, context);
   				}
   				
   			} else {
   				//global model
   				outputPorbMatrix(transProb, context);
   			}
	   	}
	   	
	   	/**
	   	 * @param transProb
	   	 * @param context
	   	 * @throws IOException
	   	 * @throws InterruptedException
	   	 */
	   	private void outputPorbMatrix(StateTransitionProbability transProb, Context context) 
	   		throws IOException, InterruptedException {
   			transProb.normalizeRows();
	   		for (int i = 0; i < states.length; ++i) {
	   			String val = transProb.serializeRow(i);
	   			outVal.set(val);
	   			context.write(NullWritable.get(),outVal);
	   		}
	   		
	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<IntWritable> values, Context context)
        		throws IOException, InterruptedException {
        	count = 0;
        	for (IntWritable value : values) {
        		count += value.get();
        	}
        	if (isClassBasedModel) {
   				//class based model
        		classLabel = key.getString(0);
            	fromSt = key.getString(1);
            	toSt = key.getString(2);
            	
            	StateTransitionProbability clsTransProb = classBasedTransProb.get(classLabel);
            	if (null == clsTransProb) {
            		clsTransProb = new StateTransitionProbability(states, states);
            		clsTransProb.setScale(transProbScale);
            		classBasedTransProb.put(classLabel, clsTransProb);
            	}
        		clsTransProb.add(fromSt, toSt, count);
        	} else {
   				//global model
        		fromSt = key.getString(0);
        		toSt = key.getString(1);
        		transProb.add(fromSt, toSt, count);
        	}
        }	   	
	}
	
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new MarkovStateTransitionModel(), args);
        System.exit(exitCode);
	}
	
	
}
