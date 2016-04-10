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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Improved Apriori algorithm for frequent item set
 * @author pranab
 *
 */
public class FrequentItemsApriori extends Configured implements Tool {
	private static final String configDelim = ",";
    private static final Logger LOG = Logger.getLogger(FrequentItemsApriori.class);

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Frequent item set with improved aPriori algorithm";
        job.setJobName(jobName);
        
        job.setJarByClass(FrequentItemsApriori.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(FrequentItemsApriori.AprioriMapper.class);
        job.setReducerClass(FrequentItemsApriori.AprioriReducer.class);
        job.setCombinerClass(FrequentItemsApriori.AprioriCombiner.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("fia.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class AprioriMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private String fieldDelimRegex;
		private String[] items;
		private int skipFieldCount;
        private Tuple outKey = new Tuple();
		private Tuple outVal  = new Tuple();
		private int itemSetLength;
        private int[] idOrdinals;
        private int tansIdOrd;
        private String transId;
        private boolean emitTransId;
        private static  final int ONE = 1;
        private ItemSetList itemSetList;
        private Set<String> currentItems = new HashSet<String>();
        private List<String> keyItems = new ArrayList<String>();
        private String infreqItemMarker;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
            skipFieldCount = conf.getInt("fia.skip.field.count", 1);
            itemSetLength = Utility.assertIntConfigParam(conf,  "fia.item.set.length",  "missing item set length");
            tansIdOrd = Utility.assertIntConfigParam(conf,  "fia.tans.id.ord",  "missing transaction id ordinal");
            emitTransId = conf.getBoolean("fia.emit.trans.id", true);
            
           	//record partition  id
        	idOrdinals = Utility.intArrayFromString(conf.get("fia.id.field.ordinals"), fieldDelimRegex);
        	
        	if (itemSetLength > 1) {
        		//load item sets of shorter length
        		itemSetList = new ItemSetList(conf, "fia.item.set.file.path", itemSetLength -1,  emitTransId, ",");
        	}
        	infreqItemMarker = conf.get("fia.infreq.item.marker");
       }

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	transId = items[tansIdOrd];
        	
        	if (1 == itemSetLength) {
        		//items sets of size 1
        		for (int i = skipFieldCount; i < items.length; ++i) {
        			outKey.initialize();
        			outVal.initialize();
        			outKey.add(items[i]);
        			if (emitTransId) {
        				outVal.add(transId);
        			} else {
            			outVal.add(ONE);
        			}
        			context.write(outKey, outVal);
        		}
        	} else {
        		//items set size greater than 1
        		if (!emitTransId) {
	        		currentItems.clear();
	        		for (int i = skipFieldCount; i < items.length; ++i) {
	        			currentItems.add(items[i]);
	        		}
        		}
        		
        		for (ItemSetList.ItemSet itemSet :  itemSetList.getItemSetList()) {
        			if (shouldGenerateLongerItemSet(itemSet)) {
        				//this transaction contained shorter length item set
                		for (int i = skipFieldCount; i < items.length; ++i) {
                			//if infrequent items marked and there is match, skip the token
                			if (null != infreqItemMarker && infreqItemMarker.equals(items[i])) {
                				continue;
                			}
                			
            				//if the item is not already contained in item set
                			if (!itemSet.containsItem(items[i])) {
                    			outKey.initialize();
                    			outVal.initialize();
                    			
                    			//existing items
                    			keyItems.clear();
                    			for (String item : itemSet.getItems()) {
                    				keyItems.add(item);
                    			}     
                    			
                    			//add new item and sort
                    			keyItems.add(items[i]);
                    			Collections.sort(keyItems);
                    			outKey.add(keyItems);
                    			
                    			if (emitTransId) {
                    				outVal.add(transId);
                    			}  else {
                        			outVal.add(ONE);
                    			}
                    			context.write(outKey, outVal);
                			}
                		}        				
        			}
        		}
        	}
        }    
        
        /**
         * @param itemSet
         * @return
         */
        private boolean shouldGenerateLongerItemSet(ItemSetList.ItemSet itemSet) {
        	boolean generate = true;
        	if (emitTransId) {
        		//only if transId found for smaller item set
        		generate = itemSet.containsTrans(transId);
        	} else {
        		//if all items of the smaller items set found in current record
        		for (String item : itemSet.getItems()) {
        			if (!currentItems.contains(item)) {
        				generate = false;
        				break;
        			}
        		}
        	}
        	
        	return generate;
        }
 	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class AprioriCombiner extends Reducer<Tuple, Tuple, Tuple, Tuple> {
		private Tuple outVal = new Tuple();
		private boolean emitTransId;
        private int transCount;
        private Set<String> transactionIds = new HashSet<String>();

		/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	protected void setup(Context context) 
	   			throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            emitTransId = conf.getBoolean("fia.emit.trans.id", true);
	   	}		
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<Tuple> values, Context context)
        		throws IOException, InterruptedException {
        	outVal.initialize();
        	transCount = 0;
        	transactionIds.clear();
        	for (Tuple value : values) {
        		if (emitTransId) {
        			//outVal.add(value);
        			for (int i = 0; i < value.getSize(); ++i) {
        				transactionIds.add(value.toString(i));
        			}
        		} else {
        			transCount += value.getInt(0);
        		}
        	}
        	if (!emitTransId) {
        		outVal.add(transCount);
        	} else {
        		for (String transID : transactionIds) {
        			outVal.add(transID);
        		}
        	}
        	context.write(key, outVal);       	
        }		
	}	
	
	
	/**
	 * @author pranab
	 *
	 */
	public static class AprioriReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private String fieldDelim;
		private Text outVal  = new Text();
		private Tuple transactionIdTuple = new Tuple();
		private Set<String> transactionIds = new HashSet<String>();
		private boolean emitTransId;
		private double supportThreshold;
		private int transCount;
        private int totalTransCount;
        private double support;
        private boolean transIdOutput;

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
            emitTransId = conf.getBoolean("fia.emit.trans.id", true);
            supportThreshold = Utility.assertDoubleConfigParam(conf, "fia.support.threshold", "missing support threshold");
            totalTransCount = Utility.assertIntConfigParam(conf,  "fia.total.tans.count",  "missing total transaction count");
            transIdOutput = conf.getBoolean("fia.trans.id.output", true);

 	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<Tuple> values, Context context)
        		throws IOException, InterruptedException {
        	transactionIds.clear();
        	transactionIdTuple.initialize();
        	transCount = 0;
        	for (Tuple value : values) {
        		if (emitTransId) {
        			for (int i = 0; i < value.getSize(); ++i) {
        				transactionIds.add(value.toString(i));
        			}
        		} else {
        			transCount += value.getInt(0);
        		}
        	}
        	
        	//emit only if support is above threshold
        	if (emitTransId) {
        		transCount = transactionIds.size();
        		for (String transID : transactionIds) {
        			transactionIdTuple.add(transID);
        		}
        	} 
        	support = (double)transCount / totalTransCount;
        	//LOG.debug("transCount=" + transCount  + "    support=" + support);
        	
        	if (support > supportThreshold) {
            	if (emitTransId) {
            		if (transIdOutput)
            			outVal.set(key.toString() + fieldDelim + transactionIdTuple.toString() + fieldDelim + Utility.formatDouble(support, 3));
            		else
            			outVal.set(key.toString() + fieldDelim +  Utility.formatDouble(support, 3));
            	} else {
            		outVal.set(key.toString() + fieldDelim + transCount + fieldDelim + Utility.formatDouble(support, 3));
            	}
       			context.write(NullWritable.get(),outVal);
        	}
        }	   	
	}	
	
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new FrequentItemsApriori(), args);
        System.exit(exitCode);
	}
	
}
