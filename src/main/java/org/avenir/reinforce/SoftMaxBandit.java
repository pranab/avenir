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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
import org.chombo.stats.RandomStringSampler;
import org.chombo.util.DynamicBean;
import org.chombo.util.Utility;

/**
 * SoftMax multi arm bandit
 * @author pranab
 *
 */
public class SoftMaxBandit    extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Soft max MAB";
        job.setJobName(jobName);
        
        job.setJarByClass(SoftMaxBandit.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(SoftMaxBandit.BanditMapper.class);
        
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

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
		private double  tempConstant;
		private String curGroupID = null;
		private String groupID;
		private int countOrdinal;
		private int rewardOrdinal;
		private Map<String, Integer> groupBatchCount = new HashMap<String, Integer>();
		private GroupedItems groupedItems = new GroupedItems();
		private RandomStringSampler sampler = new RandomStringSampler();
		private static final int DISTR_SCALE = 1000;
		
		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	fieldDelim = conf.get("field.delim", ",");

        	roundNum = conf.getInt("current.round.num",  -1);
        	tempConstant  = Double.parseDouble(conf.get("temp.constant", "1.0"));
        	countOrdinal = conf.getInt("count.ordinal",  -1);
        	rewardOrdinal = conf.getInt("reward.ordinal",  -1);
 
        	//batch size
        	List<String[]> lines = Utility.parseFileLines(conf,  "group.item.count.path",  ",");
        	String groupID;
 			int batchSize;
        	for (String[] line : lines) {
        		groupID= line[0];
        		batchSize = Integer.parseInt(line[1]);
        		groupBatchCount.put(groupID,   batchSize );
        	}
        	
        }

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void cleanup(Context context)  throws IOException, InterruptedException {
			select( context);
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            groupID = items[0];
    		if (null == curGroupID || !groupID.equals(curGroupID)) {
    			//new group
    			if (null == curGroupID) {
    				collectGroupItems();
        			curGroupID = groupID;
    			} else  {
    				//process this group
    				select( context);
    				
    				//start next group
        			groupedItems.initialize();
        			curGroupID = groupID;
    				collectGroupItems();
    			}
    		} else {
    			//existing group
				collectGroupItems();
    		}
        }

        /**
         * @return
         */
        private int getBatchSize() {
        	int batchSize = groupBatchCount.isEmpty() ? 1 : groupBatchCount.get(curGroupID);
        	return batchSize;
        }
        
        /**
         * 
         */
        private void collectGroupItems() {
        	groupedItems.createtem(items[1], Integer.parseInt(items[countOrdinal]), Integer.parseInt(items[rewardOrdinal]));
        }
        
        /**
         * @return
         * @throws InterruptedException 
         * @throws IOException 
         */
        private void select(Context context) throws IOException, InterruptedException {
        	List<String> items = new ArrayList<String>();
        	int batchSize = getBatchSize();
        	int count = (roundNum -1) * batchSize;

    		//collect items not tried before
    		List<DynamicBean> collectedItems = groupedItems.collectItemsNotTried(batchSize);
    		count += collectedItems.size();
    		for (DynamicBean it : collectedItems) {
    			items.add(it.getString(GroupedItems.ITEM_ID));
    		}
        	
    		//random sampling based on distribution 
        	sampler.initialize();
        	int maxReward = groupedItems.getMaxRewardItem().getInt(GroupedItems.ITEM_REWARD);
        	for (  DynamicBean item :  groupedItems.getGroupItems()) {
        		double distr = ((double) item.getInt(GroupedItems.ITEM_REWARD)) / maxReward;
        		int scaledDistr = (int)(Math.exp(distr /  tempConstant) * DISTR_SCALE);
        		sampler.addToDistr(item.getString(GroupedItems.ITEM_ID), scaledDistr);
        	}
        	Set<String> sampledItems = new HashSet<String>();
    		while (items.size() < batchSize) {
    			String selected = sampler.sample();
    			if (!sampledItems.contains(selected)) {
    				sampledItems.add(selected);
    				items.add(selected);
    				++count;
    			}
    		}
    		
        	//emit all selected items
          	for (String it : items) {
    			outVal.set(curGroupID + fieldDelim + it);
        		context.write(NullWritable.get(), outVal);
          	}
    		
        }        

	}

	
    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new SoftMaxBandit(), args);
        System.exit(exitCode);
    }
	
}
