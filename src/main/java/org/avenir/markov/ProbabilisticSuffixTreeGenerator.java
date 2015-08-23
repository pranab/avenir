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

        int numReducer = job.getConfiguration().getInt("pst.num.reducer", -1);
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
            skipFieldCount = conf.getInt("skip.field.count", 0);
            classLabelFieldOrd = conf.getInt("class.label.field.ord", -1);
            if (classLabelFieldOrd >= 0) {
            	++skipFieldCount;
            }
            
            treeRootSymbol = conf.get("tree.root.symbol", "$");
            maxSeqLength = conf.getInt("max.seq.length",  5);
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
        	if (items.length >= (skipFieldCount + 2)) {
        		//sequence length 2 upto max
        		for (int w = 2;  w <= maxSeqLength;  ++w ) {
        			start = skipFieldCount;
        			end = start  + w;
  
        			//sliding window
        			for (  ; end <= items.length; ++start, ++end) {
    	        		outKey.initialize();
       	        		if (null != classLabel) {
    	        			//class label based PST model
    	        			outKey.append(classLabel);
    	        		}
    	        		for (int i = start;  i < end;  ++i) {
    	        			outKey.append(items[i]);
        				}
   	        			++rootCount;
   	        		    context.write(outKey, outVal);
        			}
        		}
        	
	        	//root node
        		outKey.initialize();
        		if (null != classLabel) {
        			//class label based PST model
        			outKey.append(classLabel);
        		}
        		outKey.append(treeRootSymbol);
        		outVal.set(rootCount);
    			context.write(outKey, outVal);
	        	
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
