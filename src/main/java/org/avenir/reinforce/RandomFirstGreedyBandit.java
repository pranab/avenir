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
import java.io.InputStream;
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
		private Map<String, Integer> groupItemCounts = new HashMap<String, Integer>();
		private int roundNum;
		private int explorationCountFactor;
		private int  perRoundBatchSize;
		private int rank;
		private final int RANK_MAX = 1000;
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	roundNum = conf.getInt("current.round.num",  2);
        	explorationCountFactor = conf.getInt("exploration.count.factor",  2);
        	perRoundBatchSize = conf.getInt("per.round.batch.size",  1);
        	List<String[]> lines = Utility.parseFileLines(conf,  "group.item.count",  ",");
        	for (String[] line : lines) {
        		groupItemCounts.put(line[0], getExplorationCount(Integer.parseInt(line[1])));
        	}

        }

        private int getExplorationCount(int itemCount) {
        	return explorationCountFactor * itemCount;
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            int explorationRounds = groupItemCounts.get(items[0]) - (roundNum - 1) * perRoundBatchSize;
            outKey.initialize();
            if (explorationRounds > 0) {
            	//exploration
            	rank = (int)(Math.random() * RANK_MAX);
            } else {
            	//exploitation
            	if (items.length > 2) {
            		rank = RANK_MAX -  Integer.parseInt(items[2]);
            	} else {
            		//not explored yet
                	rank = (int)(Math.random() * RANK_MAX);
            	}
            }
        	outKey.add(items[0], rank);
        	outVal.set(items[1]);
          	context.write(outKey, outVal);
       }
 	}	
	
	
	/**
	 * @author pranab
	 *
	 */
	public static class BanditReducer extends Reducer<Tuple, Text, NullWritable, Text> {
		private Text valOut;
		private String fieldDelim;
		
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
        	for (Text value : values) {
        		if (first) {
        			val = value.toString();;
        			break;
        		}
        	}
        	
        	valOut.set(key.getString(0) + fieldDelim + val);
			context.write(NullWritable.get(), valOut);
        }	   	
	}	
}
