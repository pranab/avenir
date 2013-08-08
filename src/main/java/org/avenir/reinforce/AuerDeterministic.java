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
		
		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	fieldDelim = conf.get("field.delim", ",");

        	roundNum = conf.getInt("current.round.num",  -1);
        	detAlgorithm = conf.get("det.algorithm", AUER_DET_UBC1 );
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
    			} else  {
    				select( context);
    			}
    			
    			groupedItems.initialize();
    			curGroupID = groupID;
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
			 if (detAlgorithm.equals(AUER_DET_UBC1 )) {
				 deterministicAuerSelect(context);
			} 
        }        
        
        
        /**
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void deterministicAuerSelect(Context context) throws IOException, InterruptedException {
        	List<String> items = new ArrayList<String>();
        	int batchSize = getBatchSize();
        	int count = (roundNum -1) * batchSize;
    		int maxReward = 0;
    		String item = null;
			int thisCount;
    		int reward;

    		//max reward in this group
    		DynamicBean maxRewardItem = groupedItems.getMaxRewardItem();
    		maxReward = maxRewardItem.getInt(ITEM_REWARD);

    		//collect items not tried before
    		List<DynamicBean> collectedItems = groupedItems.collectItemsNotTried(batchSize);
    		count += collectedItems.size();
    		for (DynamicBean it : collectedItems) {
    			items.add(it.getString(ITEM_ID));
    		}
    		
    		//collect items according to UBC 
    		while (items.size() < batchSize) {
    			double valueMax = 0.0;
    			double value;
    			DynamicBean selectedGroupItem = null;
    			List<DynamicBean> groupItems = groupedItems.getGroupItems();
        		for (DynamicBean groupItem : groupItems) {
        			reward = groupItem.getInt(ITEM_REWARD);
        			thisCount = groupItem.getInt(ITEM_COUNT);
        			value = ((double)reward) / maxReward  +   Math.sqrt(2.0 * Math.log(count) / thisCount);
        			if (value > valueMax) {
        				item = groupItem.getString(ITEM_ID);
        				valueMax = value;
        				selectedGroupItem = groupItem;
        			}
        		}
        		
				items.add(item);
				groupItems.remove(selectedGroupItem);
				++count;
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
        int exitCode = ToolRunner.run(new AuerDeterministic(), args);
        System.exit(exitCode);
    }
}
