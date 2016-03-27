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
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
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
 * Builds HMM from labeled data. Data could be fully or partially tagged
 * @author pranab
 *
 */
public class HiddenMarkovModelBuilder extends Configured implements Tool {
	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "HMM model builder";
        job.setJobName(jobName);
        
        job.setJarByClass(HiddenMarkovModelBuilder.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(HiddenMarkovModelBuilder.StateTransitionMapper.class);
        job.setReducerClass(HiddenMarkovModelBuilder.StateTransitionReducer.class);
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
		private List<String[]> obsStateList = new ArrayList<String[]>();
		private String subFieldDelim;
		private boolean partiallyTagged;
		private String[] states;
		private int[] windowFunction;
		private List<Integer> stateIndexes = new ArrayList<Integer>();
		private static Integer STATE_TRANS = 0;
		private static Integer STATE_OBS = 1;
		private static Integer INITIAL_STATE = 2;
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
            skipFieldCount = conf.getInt("hmmb.skip.field.count", 0);
        	subFieldDelim = conf.get("sub.field.delim", ":");
        	partiallyTagged = conf.getBoolean("hmmb.partially.tagged", false);
        	states = conf.get("hmmb.model.states").split(",");
        	if (partiallyTagged) {
        		windowFunction = Utility.intArrayFromString(conf.get("hmmb.window.function"), ",");
        	}
        }
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	if (partiallyTagged) {
        		processPartiallyTagged(context);
        	} else {
        		processFullyTagged(context);
        	}
        }        
        
        /**
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void processFullyTagged(Context context) throws IOException, InterruptedException {
        	obsStateList.clear();
        	if (items.length >= (skipFieldCount + 2)) {
	        	for (int i = skipFieldCount; i < items.length; ++i) {
	        		String[] obsState = items[i].split(subFieldDelim);;
	        		obsStateList.add(obsState);
	        	}
        	}
        	
        	//all observation state pairs
        	for (int i = 0; i < obsStateList.size(); ++i) {
        		if (i == 0) {
        			//intial state
	        		outKey.initialize();
	        		outKey.add(INITIAL_STATE, obsStateList.get(i)[1], obsStateList.get(i)[1]);
	        		context.write(outKey, outVal);
        		}
        		
        		//state observation
        		outKey.initialize();
        		outKey.add(STATE_OBS, obsStateList.get(i)[1], obsStateList.get(i)[0]);
        		context.write(outKey, outVal);
        		
        		if (i > 0) {
        			//state transition
            		outKey.initialize();
            		outKey.add(STATE_TRANS, obsStateList.get(i-1)[1], obsStateList.get(i)[1]);
            		context.write(outKey, outVal);
        		}
        	}
        }
        
        
        /**
         * @param context
         * @throws InterruptedException 
         * @throws IOException 
         */
        private void processPartiallyTagged(Context context) throws IOException, InterruptedException {
        	//identify states
        	stateIndexes.clear();
        	for (int i = 0; i <  items.length; ++i) {
        		if (ArrayUtils.contains(states, items[i])) {
        			stateIndexes.add(i);
        		}
        	}

 			//intial state
    		outKey.initialize();
    		outKey.add(INITIAL_STATE, items[stateIndexes.get(0)], items[stateIndexes.get(0)]);
    		outVal.set(1);
    		context.write(outKey, outVal);
 
    		//state to observation 
        	int leftBound = 0;
        	int rightBound = 0;
        	int leftWindow = 0;
        	int rightWindow = 0;
        	for (int i = 0; i < stateIndexes.size(); ++i ) {
        		//boundary on left
        		if (i > 0) {
        			leftWindow = stateIndexes.get(i) - stateIndexes.get(i-1) / 2;
        			leftBound =stateIndexes.get(i) - leftWindow; 
        		} else {
        			leftBound = -1;
        		}
        		
        		//boundary on right
        		if (i  < stateIndexes.size() -1) {
        			rightWindow = stateIndexes.get(i+1) - stateIndexes.get(i) / 2; 
        			rightBound =  stateIndexes.get(i) + rightWindow;
        		} else {
        			rightBound = -1;
        		}
        		
        		//at ends
        		if (leftBound == -1 && rightBound != -1) {
        			//first state
        			leftBound =stateIndexes.get(i) - rightWindow; 
        			if (leftBound < 0) {
        				leftBound = 0;
        			}
        		} else if    (rightBound == -1 && leftBound != -1) {
        			//last state
        			rightBound =stateIndexes.get(i) + leftWindow; 
        			if (rightBound >= items.length) {
        				rightBound =  items.length-1;
        			}
        		} else if (leftBound == -1 && rightBound == -1) {
        			//only one state
        			leftBound =stateIndexes.get(i) / 2;
        			rightBound = stateIndexes.get(i) + (items.length - 1 - stateIndexes.get(i)) / 2;
        		}
        		
        		//state observation count to left
        		String state = items[stateIndexes.get(i)];
        		for (int j = stateIndexes.get(i)-1, k=0;  j >= leftBound;  --j,++k ) {
        			String obs = items[j];
            		outKey.initialize();
            		outKey.add(STATE_OBS, state, obs);
            		int val = k < windowFunction.length ?  windowFunction[k]  :   windowFunction[ windowFunction.length -1];  
            		outVal.set(val);
            		context.write(outKey, outVal);
        		}
        		
           		//state observation count to left
        		for (int j = stateIndexes.get(i)+1, k=0;  j <= rightBound;  ++j,++k ) {
        			String obs = items[j];
            		outKey.initialize();
            		outKey.add(STATE_OBS, state, obs);
            		int val = k < windowFunction.length ?  windowFunction[k]  :   windowFunction[ windowFunction.length -1];  
            		outVal.set(val);
            		context.write(outKey, outVal);
        		}
        		
        	}
        	
        	//state to state
        	for (int i = 0;  i < stateIndexes.size() -1; ++i) {
        		outKey.initialize();
        		outKey.add(STATE_TRANS, items[stateIndexes.get(i)],  items[stateIndexes.get(i+1)]);
        		outVal.set(1);
        		context.write(outKey, outVal);
        	}
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
		private String[] observations;
		private String[] initial;
		private StateTransitionProbability stateTransProb;
		private StateTransitionProbability stateObsProb;
		private StateTransitionProbability initialStateProb;
		private int count;
		private static Integer STATE_TRANS = 0;
		private static Integer STATE_OBS = 1;
		private static Integer INITIAL_STATE = 2;
		private static String INITIAL = "initial";
		private static final Logger LOG = Logger.getLogger(StateTransitionMapper.class);
		
	   	protected void setup(Context context) 
	   			throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelim = conf.get("field.delim.out", ",");
        	states = conf.get("hmmb.model.states").split(",");
        	observations = conf.get("hmmb.model.observations").split(",");
        	int transProbScale = conf.getInt("hmmb.trans.prob.scale", 1000);
        	
        	//state transition
        	stateTransProb = new StateTransitionProbability(states, states);
        	stateTransProb.setScale(transProbScale);
        	
        	//state observation
        	stateObsProb = new StateTransitionProbability(states, observations);
        	stateObsProb.setScale(transProbScale);
        	
        	//initial state
        	initial = new String[1];
        	initial[0] = INITIAL;
        	initialStateProb = new StateTransitionProbability(initial, states);
	   	}
	   	
	   	protected void cleanup(Context context)  
	   			throws IOException, InterruptedException {
	   		//all states
	   		outVal.set(Utility.join(states));
   			context.write(NullWritable.get(),outVal);
	   		
	   		//all observations
	   		outVal.set(Utility.join(observations));
   			context.write(NullWritable.get(),outVal);
	   		
	   		
	   		//state transition
	   		stateTransProb.normalizeRows();
	   		for (int i = 0; i < states.length; ++i) {
	   			String val = stateTransProb.serializeRow(i);
	   			outVal.set(val);
	   			context.write(NullWritable.get(),outVal);
	   		}
	   		
	   		//state observation
	   		stateObsProb.normalizeRows();
	   		for (int i = 0; i < states.length; ++i) {
	   			String val = stateObsProb.serializeRow(i);
	   			outVal.set(val);
	   			context.write(NullWritable.get(),outVal);
	   		}
	   		
	   		
	   		//intial state
	   		initialStateProb.normalizeRows();
   			String val = initialStateProb.serializeRow(0);
   			outVal.set(val);
   			context.write(NullWritable.get(),outVal);
	   		
	   	}
	   	
        protected void reduce(Tuple  key, Iterable<IntWritable> values, Context context)
        		throws IOException, InterruptedException {
        	count = 0;
        	for (IntWritable value : values) {
        		count += value.get();
        	}
        	
        	//state transition
        	if (key.getInt(0) == STATE_TRANS) {
        		String fromSt = key.getString(1);
        		String toSt = key.getString(2);
        		stateTransProb.add(fromSt, toSt, count);
        	} else if (key.getInt(0) == STATE_OBS) { 
        		String fromSt = key.getString(1);
        		String toObs = key.getString(2);
        		stateObsProb.add(fromSt, toObs, count);
        	} else if (key.getInt(0) == INITIAL_STATE) {
        		String toSt = key.getString(1);
        		initialStateProb.add(INITIAL, toSt, count);
        	}
        }	   	
	}
	
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new HiddenMarkovModelBuilder(), args);
        System.exit(exitCode);
	}
	
}
