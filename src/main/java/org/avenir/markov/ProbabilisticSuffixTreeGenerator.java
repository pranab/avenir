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
import java.util.HashMap;
import java.util.List;
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
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Generated higher order conditional probability for sequence data. Can be used for higher order
 * Markov Model
 * @author pranab
 *
 */
public class ProbabilisticSuffixTreeGenerator extends Configured implements Tool {
	private static final String configDelim = ",";

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Higher conditional probability for sequence data with probabilistic suffix tree";
        job.setJobName(jobName);
        
        job.setJarByClass(ProbabilisticSuffixTreeGenerator.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(ProbabilisticSuffixTreeGenerator.SuffixTreeMapper.class);
        job.setReducerClass(ProbabilisticSuffixTreeGenerator.SuffixTreeReducer.class);
        job.setCombinerClass(ProbabilisticSuffixTreeGenerator.SuffixTreeCombiner.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("pstg.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class SuffixTreeMapper extends Mapper<LongWritable, Text, Tuple, IntWritable> {
		private String fieldDelimRegex;
		private String[] items;
		private int skipFieldCount;
        private Tuple outKey = new Tuple();
		private IntWritable outVal  = new IntWritable(1);
		private int classLabelFieldOrd;
		private String classLabel;
		private String treeRootSymbol;
		private int rootCount;
		private int maxSeqLength;
		private int seqLength;
		private int start;
		private int end;
        private int[] idOrdinals;
        private boolean inputFormatSequential;
        private List<String> window = new  ArrayList<String>();
        private Map<String, List<String>> partitionedWindows = new HashMap<String, List<String>>();
        private int dataFieldOrdinal;
        private static final Logger LOG = Logger.getLogger(SuffixTreeMapper.class);

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
            skipFieldCount = conf.getInt("pstg.skip.field.count", 0);
            classLabelFieldOrd = conf.getInt("pstg.class.label.field.ord", -1);
            if (classLabelFieldOrd >= 0) {
            	++skipFieldCount;
            }
            
            treeRootSymbol = conf.get("pstg.tree.root.symbol", "$");
            maxSeqLength = conf.getInt("pstg.max.seq.length",  5);
            
           	//record partition  id
        	idOrdinals = Utility.intArrayFromString(conf.get("pstg.id.field.ordinals"), fieldDelimRegex);
            
        	inputFormatSequential = conf.getBoolean("pstg.input.format.sequential", true);
        	if (!inputFormatSequential) {
        		dataFieldOrdinal = Utility.assertIntConfigParam(conf, "pstg.data.field.ordinal",  
        				"for non sequential data data field ordinal must be specified");
        	}
        }
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	rootCount = 0;
        	
    		if (classLabelFieldOrd >= 0) {
    			//class label based markov model
    			classLabel = items[classLabelFieldOrd];
    		}

    		//generate suffix nodes
    		if (inputFormatSequential) {
	        	if (items.length >= (skipFieldCount + 2)) {
	        		//sequence length 2 up to max
	        		for (int w = 2;  w <= maxSeqLength;  ++w ) {
	        			start = skipFieldCount;
	        			end = start  + w;
	  
	        			//sliding window
	        			for (  ; end <= items.length; ++start, ++end) {
	    	        		outKey.initialize();
	    	        		
	    	        		//partition id
	    	        		outKey.addFromArray(items, idOrdinals);
	    	        		
		        			//class label based PST model
	       	        		if (null != classLabel) {
	    	        			outKey.append(classLabel);
	    	        		}
	    	        		for (int i = start;  i < end;  ++i) {
	    	        			outKey.append(items[i]);
	        				}
	   	        			++rootCount;
	   	        		    context.write(outKey, outVal);
	        			}
	        	}
        	} else {
        			if (null != idOrdinals) {
    					String compId = Utility.join(items, 0, idOrdinals.length, configDelim);
    					List<String> partWindow = partitionedWindows.get(compId);
    					if (null == partWindow) {
    						partWindow =new  ArrayList<String>();
    						partitionedWindows.put(compId, partWindow);
    					}
    					updateWindowAndEmit(partWindow, context);
        			} else {
        				updateWindowAndEmit(window, context);
        			}
        		}
	        	
	        	
	        	//root node
	        	emitRoot( context);
    		}
        }
        
        /**
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void emitRoot(Context context) throws IOException, InterruptedException {
        	//root node
    		outKey.initialize();
    		outKey.addFromArray(items, idOrdinals);
    		if (null != classLabel) {
    			//class label based PST model
    			outKey.append(classLabel);
    		}
    		outKey.append(treeRootSymbol);
    		outVal.set(rootCount);
			context.write(outKey, outVal);
        }
        
	/**
	 * @param window
	 * @param context
	 * @throws IOException
	 * @throws InterruptedException
	 */
	private void updateWindowAndEmit(List<String> window, Context context) throws IOException, InterruptedException {
		window.add(items[dataFieldOrdinal]);
		if (window.size() > maxSeqLength) {
			window.remove(0);
		}
		if (window.size() == maxSeqLength) {
    		for (int w = 2;  w <= maxSeqLength;  ++w ) {
    			end =  w;
        		outKey.initialize();
        		
        		//partition Id
        		outKey.addFromArray(items, idOrdinals);
    			
        		//class label based PST model
	        	if (null != classLabel) {
        			outKey.append(classLabel);
        		}
        		for (int i = 0; i < end; ++i) {
        			outKey.add(window.get(i));
        		}
       		    context.write(outKey, outVal);
       		    ++rootCount;
    		}        			
		}
	}
	
}	
       
	/**
	 * @author pranab
	 *
	 */
	public static class SuffixTreeCombiner extends Reducer<Tuple, IntWritable, Tuple, IntWritable> {
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
	public static class SuffixTreeReducer extends Reducer<Tuple, IntWritable, NullWritable, Text> {
		private String fieldDelim;
		private Text outVal  = new Text();
		private int count;
		private static final Logger LOG = Logger.getLogger(SuffixTreeReducer.class);

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
        	
        	key.setDelim(fieldDelim);
        	outVal.set(key.toString() + fieldDelim + count );
   			context.write(NullWritable.get(),outVal);
        }	   	
	}	
	
	
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new ProbabilisticSuffixTreeGenerator(), args);
        System.exit(exitCode);
	}
	
}
