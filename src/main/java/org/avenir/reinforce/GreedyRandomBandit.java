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

package org.avenir.reinforce;

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
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.util.Pair;
import org.chombo.util.Utility;

/**
 * Implements greedy multiarm bandit  reinforcement learning algorithms
 * @author pranab
 *
 */
public class GreedyRandomBandit   extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Greedy random bandit problem";
        job.setJobName(jobName);
        
        job.setJarByClass(GreedyRandomBandit.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(GreedyRandomBandit.BanditMapper.class);
        
        job.setMapOutputKeyClass(NullWritable.class);
        job.setMapOutputValueClass(Text.class);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class BanditMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
		private String fieldDelimRegex;
    	private String fieldDelim ;
		private String[] items;
		private Text outVal  = new Text();
		private int roundNum;
		private float randomSelectionProb;
		private String  probRedAlgorithm;
		private String curGroupID = null;
		private String groupID;
		private int rewardOrdinal;
		private List<Pair<String, Integer>> groupItems = new ArrayList<Pair<String, Integer>>();
		private static final String PROB_RED_LINEAR = "linear";
		private static final String PROB_RED_LOG_LINEAR = "logLinear";
		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	fieldDelim = conf.get("field.delim", ",");

        	roundNum = conf.getInt("current.round.num",  2);
        	randomSelectionProb = conf.getFloat("random.selection.prob", (float)0.5);
        	probRedAlgorithm = conf.get("prob.reduction.algorithm", PROB_RED_LINEAR );
        	rewardOrdinal = conf.getInt("reward.ordinal",  -1);
        }

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void cleanup(Context context)  throws IOException, InterruptedException {
			if (probRedAlgorithm.equals(PROB_RED_LINEAR )) {
				 linearSelect(context);
			}	else if (probRedAlgorithm.equals(PROB_RED_LOG_LINEAR )) {
				 logLinearSelect(context);
			}	
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            groupID = items[0];
    		if (null == curGroupID || !groupID.equals(curGroupID)) {
    			//new group
    			if (null == curGroupID) {
        			Pair<String, Integer> item = new Pair<String, Integer>(items[1], Integer.parseInt(items[rewardOrdinal])); 
        			groupItems.add(item);
    			} else  {
	    			if (probRedAlgorithm.equals(PROB_RED_LINEAR )) {
	    				 linearSelect(context);
	    			} else if (probRedAlgorithm.equals(PROB_RED_LOG_LINEAR )) {
	    				 logLinearSelect(context);
	    			}
    			}
    			
    			groupItems.clear();
    			curGroupID = groupID;
    		} else {
    			//existing group
    			Pair<String, Integer> item = new Pair<String, Integer>(items[1], Integer.parseInt(items[rewardOrdinal])); 
    			groupItems.add(item);
    		}
        }
        
        /**
         * @return
         * @throws InterruptedException 
         * @throws IOException 
         */
        private void linearSelect(Context context) throws IOException, InterruptedException {
        	float curProb = randomSelectionProb / roundNum;
        	linearSelectHelper(curProb, context);
        }
        
        /**
         * @return
         * @throws InterruptedException 
         * @throws IOException 
         */
        private void logLinearSelect(Context context) throws IOException, InterruptedException {
        	float curProb = (float)(randomSelectionProb * Math.log(roundNum) / roundNum);
        	linearSelectHelper(curProb, context);
        }

        /**
         * @param curProb
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void linearSelectHelper(float curProb, Context context) throws IOException, InterruptedException {
        	String itemID = null;
        	if (curProb < Math.random()) {
        		//random
        		int select = (int)Math.round( Math.random() * groupItems.size());
        		itemID = groupItems.get(select).getLeft();
        	} else {
        		//choose best so far
        		int maxReward = 0;
        		for (Pair<String, Integer> groupItem : groupItems) {
        			if (groupItem.getRight() > maxReward) {
        				itemID = groupItem.getLeft();
        			}
        		}
        	}
			outVal.set(curGroupID + fieldDelim + itemID);
    		context.write(NullWritable.get(), outVal);
        }
        
        
	}
	
	
    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new GreedyRandomBandit(), args);
        System.exit(exitCode);
    }
	
}
