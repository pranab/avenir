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
import java.util.HashSet;
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
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.chombo.util.Utility;

/**
 * Replaces infrequent items with a marker. Makes aPriori faster. Should be run after frequent 1 item
 * sets are found 
 * @author pranab
 *
 */
public class InfrequentItemMarker extends Configured implements Tool {
	private static final String configDelim = ",";
    private static final Logger LOG = Logger.getLogger(FrequentItemsApriori.class);

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Marks infrequent items with with a mask string";
        job.setJobName(jobName);
        
        job.setJarByClass(InfrequentItemMarker.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(InfrequentItemMarker.MarkerMapper.class);
        
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setNumReduceTasks(0);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class MarkerMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
		private String fieldDelimRegex;
		private String fieldDelimOut;
		private String[] items;
		private int skipFieldCount;
		private Text outVal  = new Text();
		private int itemSetLength;
		private boolean containsTransId;
        private ItemSetList itemSetList;
        private String infreqItemMarker;
        private Set<String> freqItems = new HashSet<String>();
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
            if (config.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = Utility.getFieldDelimiter(config, "iim.field.delim.regex", "field.delim.regex", ",");
        	fieldDelimOut = Utility.getFieldDelimiter(config, "iim.field.delim.out", "field.delim.out", ",");
        	
            skipFieldCount = config.getInt("iim.skip.field.count", 1);
            itemSetLength = Utility.assertIntConfigParam(config,  "iim.item.set.length",  "missing item set length");
            if (1 != itemSetLength) {
            	throw new IllegalStateException("expecting item set of length 1");
            }
            containsTransId = config.getBoolean("iim.contains.trans.id", true);

        	//infreq item marker
        	infreqItemMarker = config.get("iim.infreq.item.marker", "*");
        	
        	//load item sets of length 1
        	itemSetList = new ItemSetList(config, "iim.item.set.file.path", itemSetLength,  containsTransId, 
        			config.get("iim.itemset.delim",","));
        	
        	//hash of frequent items
        	for (ItemSetList.ItemSet itemSet :  itemSetList.getItemSetList()) {
        		if (itemSet.getItems().size() != 1) {
                	throw new IllegalStateException("expecting item set of length 1");
        		}
        		for (String item : itemSet.getItems()) {
        			freqItems.add(item);
        		}
        	}
        }
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	
    		for (int i = skipFieldCount; i < items.length; ++i) {
    			if (!freqItems.contains(items[i])) {
    				items[i]  = infreqItemMarker;
    			}
    		}
    		
    		outVal.set(Utility.join(items, fieldDelimOut));
   			context.write(NullWritable.get(),outVal);
        }        
	}
	
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new InfrequentItemMarker(), args);
        System.exit(exitCode);
	}
	

}
