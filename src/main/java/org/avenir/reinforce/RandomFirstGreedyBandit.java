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
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.util.SecondarySort;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Random first greedy later bandit reinforcement learning
 * @author pranab
 *
 */
public class RandomFirstGreedyBandit   extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Random first greedy  bandit problem";
        job.setJobName(jobName);
        
        job.setJarByClass(RandomFirstGreedyBandit.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(RandomFirstGreedyBandit.BanditMapper.class);
        job.setReducerClass(RandomFirstGreedyBandit.BanditReducer.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Text.class);

        job.setGroupingComparatorClass(SecondarySort.TuplePairGroupComprator.class);
        job.setPartitionerClass(SecondarySort.TupleTextPartitioner.class);
        
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
	public static class BanditMapper extends Mapper<LongWritable, Text, Tuple, Text> {
		private String fieldDelimRegex;
		private String[] items;
		private Text outVal  = new Text();
		private Tuple outKey = new  Tuple();
		private int roundNum;
		private int explorationCountFactor;
		private int rank;
		private final int RANK_MAX = 1000;
		private Map<String, ExplorationCounter> explCounters = new HashMap<String, ExplorationCounter>();
		private String curGroupID = null;
		private String groupID;
		private ExplorationCounter curExplCounter;
		private int  curItemIndex = 0;
		private String explCountStrategy;
		private static final String EXPL_STRATEGY_SIMPLE = "simple";
		private static final String EXPL_STRATEGY_PAC = "pac";
		private float rewardDiff;
		private float probDiff ;
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	roundNum = conf.getInt("current.round.num",  2);
        	explCountStrategy = conf.get("exploration.count.strategy", EXPL_STRATEGY_SIMPLE);
        	if (explCountStrategy.equals(EXPL_STRATEGY_SIMPLE)) {
            	explorationCountFactor = conf.getInt("exploration.count.factor",  2);
        	} else {
        		rewardDiff = conf.getFloat("pac.reward.diff", (float)0.2);
        		probDiff = conf.getFloat("pac.prob.diff", (float)0.2);
        	}
        	List<String[]> lines = Utility.parseFileLines(conf,  "group.item.count.path",  ",");
        	
        	String groupID;
        	int count; 
        	int explorationCount;
			int batchSize;
        	for (String[] line : lines) {
        		groupID= line[0];
        		count = Integer.parseInt(line[1]);
        		batchSize = Integer.parseInt(line[2]);
        		explorationCount = getExplorationCount(count);
        		explCounters.put(groupID, new ExplorationCounter( groupID,  count,  explorationCount,  batchSize) );
        	}

        }

        /**
         * calculates exploration count
         * @param itemCount
         * @return
         */
        private int getExplorationCount(int itemCount) {
        	int explCount = 0;
        	if (explCountStrategy.equals(EXPL_STRATEGY_SIMPLE)) {
        		explCount =  explorationCountFactor * itemCount;
        	} else {
        		explCount = (int)(4.0 / (rewardDiff * rewardDiff) + Math.log( 2.0 * itemCount / probDiff));
        	}
        	
        	return explCount;
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            groupID = items[0];
    		if (null == curGroupID || !groupID.equals(curGroupID)) {
    			//new group
    			curExplCounter = explCounters.get(groupID);
    			curExplCounter.selectNextRound(roundNum);
    			curGroupID= groupID;
    			curItemIndex = 0;
    		} else {
    			//same group
    			++curItemIndex;
    		}
    		
            outKey.initialize();
            if (curExplCounter.isInExploration()) {
            	//exploration
                if (curExplCounter.shouldExplore(curItemIndex)) {
                	rank = 1;
                } else {
                	rank = -1;
                }
            } else {
            	//exploitation
            	if (items.length > 2) {
            		rank = RANK_MAX -  Integer.parseInt(items[2]);
            	}  else {
            		rank = -1;
            	}
            }
            
            //emit if current item needs exploration or current items needs to be exploited and reawrd data is available
            if (rank > 0) {
            	if (0 == curItemIndex) {
            		//if new group emit batch size
            		outKey.add(items[0], -1);
            		outVal.set("" + curExplCounter.getBatchSize());
                	context.write(outKey, outVal);
                    outKey.initialize();
            	}
            	
            	//emit rank
            	outKey.add(items[0], rank);
            	outVal.set(items[1]);
            	context.write(outKey, outVal);
            }
       }
 	
	}	
	
	
	/**
	 * @author pranab
	 *
	 */
	public static class BanditReducer extends Reducer<Tuple, Text, NullWritable, Text> {
		private Text valOut;
		private String fieldDelim;
		
	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	fieldDelim = conf.get("field.delim", ",");
	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<Text> values, Context context)
        throws IOException, InterruptedException {
        	boolean first = true;
        	String val = null;
        	int batchCount = 0;
        	int batchSize = 0;
        	String groupID = key.getString(0);
        	for (Text value : values) {
        		if (first) {
        			//first one is batch size
        			val = value.toString();
        			batchSize = Integer.parseInt(val);
        			first = false;
        		} else {
        			//select as many as batch size
        			val = value.toString();
                	valOut.set(groupID + fieldDelim + val);
        			context.write(NullWritable.get(), valOut);
        			if (++batchCount == batchSize) {
        				break;
        			}
        		}
        	}
        }	   	
	}	
	
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new RandomFirstGreedyBandit(), args);
        System.exit(exitCode);
    }
	
}
