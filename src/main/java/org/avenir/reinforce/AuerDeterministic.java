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
import org.chombo.util.BasicUtils;
import org.chombo.util.DynamicBean;
import org.chombo.util.Utility;

/**
 * Deterministic Auer MAB algorithm
 * @author pranab
 *
 */
public class AuerDeterministic  extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Auer determininstic MAB";
        job.setJobName(jobName);
        
        job.setJarByClass(AuerDeterministic.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(AuerDeterministic.BanditMapper.class);
        
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
		private String  detAlgorithm;
		private String curGroupID = null;
		private String groupID;
		private int countOrdinal;
		private int rewardOrdinal;
		private static final String AUER_DET_UBC1 = "AuerUBC1";
		private static final String ITEM_ID = "itemID";
		private static final String ITEM_COUNT = "count";
		private static final String ITEM_REWARD = "reward";
		private Map<String, Integer> groupBatchCount = new HashMap<String, Integer>();
		private GroupedItems groupedItems = new GroupedItems();
		private int globalBatchSize;
		private int minReward;
		private boolean outputDecisionCount;
		private static final int GR_ORD = 0;
		private static final int IT_ORD = 1;
		
		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	fieldDelim = conf.get("field.delim", ",");

        	roundNum = Utility.assertIntConfigParam(conf, "current.round.num", "missing round number config param");
        	detAlgorithm = conf.get("det.algorithm", AUER_DET_UBC1 );
        	countOrdinal = conf.getInt("count.ordinal",  -1);
        	rewardOrdinal = conf.getInt("reward.ordinal",  -1);
        	minReward = conf.getInt("min.reward",  5);
        	outputDecisionCount = conf.getBoolean("output.decision.count", false);
        	
        	//batch size
        	globalBatchSize = conf.getInt("global.batch.size", -1);
        	if (globalBatchSize < 0) {
        		List<String[]> lines = Utility.parseFileLines(conf,  "group.item.count.path",  ",");
	        	if (lines.isEmpty()) {
	        		throw new IllegalStateException("either global batch size or groupwise batch size needs to be defined");
	        	}
        		String groupID;
 				int batchSize;
        		for (String[] line : lines) {
        			groupID= line[0];
        			batchSize = Integer.parseInt(line[1]);
        			groupBatchCount.put(groupID,   batchSize );
        		}
        	}
        }

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void cleanup(Context context)  throws IOException, InterruptedException {
        	super.cleanup(context);
			select(context);
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            groupID = items[GR_ORD];
    		if (null == curGroupID || !groupID.equals(curGroupID)) {
    			//new group
    			if (null == curGroupID) {
    				//first group
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
        	int batchSize = groupBatchCount.isEmpty() ? globalBatchSize : groupBatchCount.get(curGroupID);
        	return batchSize;
        }
        
        /**
         * create item under a group
         */
        private void collectGroupItems() {
        	int count = Integer.parseInt(items[countOrdinal]);
        	int reward = BasicUtils.roundToInt(Double.parseDouble(items[rewardOrdinal]));
        	groupedItems.createtem(items[IT_ORD], count, reward);
        }
        
        /**
         * @return
         * @throws InterruptedException 
         * @throws IOException 
         */
        private void select(Context context) throws IOException, InterruptedException {
			 if (detAlgorithm.equals(AUER_DET_UBC1 )) {
				 deterministicAuerSelect(context);
			} else {
				throw new IllegalArgumentException("inalid auer deterministic algorithm");
			}
        }        
        
        
        /**
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void deterministicAuerSelect(Context context) throws IOException, InterruptedException {
        	List<String> selItems = new ArrayList<String>();
        	int batchSize = getBatchSize();
        	int count = (roundNum -1) * batchSize;

			//collect items not tried before
			count = collectUntriedItems(selItems, batchSize, count);
	
			//collect items according to UBC 
			count = collectItemsByValue(selItems, batchSize, count);
        	
        	//emit all selected items
			if (outputDecisionCount) {
				Map<String, Integer> decisionCount = new HashMap<String, Integer>();
	          	for (String item : selItems) {
	          		Integer itemCount = decisionCount.get(item);
	          		if (null == itemCount) {
	          			itemCount = 0;
	          		} 
	          		decisionCount.put(item, itemCount + 1);
	          	}
	          	for (String item : decisionCount.keySet()) {
	    			outVal.set(curGroupID + fieldDelim + item + fieldDelim + decisionCount.get(item));
	        		context.write(NullWritable.get(), outVal);
	          	}
			} else {
				for (String item : selItems) {
					outVal.set(curGroupID + fieldDelim + item);
					context.write(NullWritable.get(), outVal);
				}
			}
        }

        /**
         * @param items
         * @param batchSize
         * @return
         */
        private int collectUntriedItems(List<String> items, int batchSize, int count) {
			List<DynamicBean> collectedItems = groupedItems.collectItemsNotTried(batchSize);
			count += collectedItems.size();
			for (DynamicBean it : collectedItems) {
				items.add(it.getString(ITEM_ID));
				if (minReward > 0) {
					groupedItems.select(it, minReward);
				}
			}
			return count;
        }
        
        /**
         * @param items
         * @param batchSize
         * @return
         */
        private int collectItemsByValue(List<String> items, int batchSize, int count) {
    		int maxReward = 0;
    		String item = null;
			int thisCount = 0;
    		int reward = 0;
    		
			while (items.size() < batchSize) {
				//max reward in this group
				DynamicBean maxRewardItem = groupedItems.getMaxRewardItem();
				maxReward = maxRewardItem.getInt(ITEM_REWARD);
		
				double valueMax = 0.0;
				double value;
				DynamicBean selectedGroupItem = null;
				List<DynamicBean> groupItems = groupedItems.getGroupItems();
				for (DynamicBean groupItem : groupItems) {
					reward = groupedItems.getReward(groupItem);
					thisCount = groupedItems.getTotalCount(groupItem);
					if (thisCount > 0) {
						value = ((double)reward) / maxReward  +   Math.sqrt(2.0 * Math.log(count) / thisCount);
						if (value > valueMax) {
							item = groupItem.getString(ITEM_ID);
							valueMax = value;
							selectedGroupItem = groupItem;
						}
					}
				}
		
				if (null != selectedGroupItem) {
					items.add(item);
					if (minReward > 0) {
						groupedItems.select(selectedGroupItem, minReward);
					}
					++count;
				} else {
					throw new IllegalArgumentException("Should not be here. Failed to select item by value");
				}
			}
        	return count;
        }
	}	
	
    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new AuerDeterministic(), args);
        System.exit(exitCode);
    }
}
