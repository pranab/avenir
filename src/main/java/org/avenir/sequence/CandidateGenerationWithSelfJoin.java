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

package org.avenir.sequence;

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
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.chombo.util.SecondarySort;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Generates k candidate sequence by self joining k-1 frequent sequence. Used by GSP sequence
 * clustering
 * @author pranab
 *
 */
public class CandidateGenerationWithSelfJoin extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(CandidateGenerationWithSelfJoin.class);

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Generates k candidate sequence";
        job.setJobName(jobName);
        
        job.setJarByClass(CandidateGenerationWithSelfJoin.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(CandidateGenerationWithSelfJoin.CandidateGenerationMapper.class);
        job.setReducerClass(CandidateGenerationWithSelfJoin.CandidateGenerationReducer.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setGroupingComparatorClass(SecondarySort.TuplePairGroupComprator.class);
        job.setPartitionerClass(SecondarySort.TuplePairPartitioner.class);
        
        int numReducer = job.getConfiguration().getInt("cgs.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class CandidateGenerationMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private String fieldDelimRegex;
		private String[] items;
        private Tuple outKey = new Tuple();
		private Tuple outVal  = new Tuple();
		private int itemSetLength;
        private int bucketCount;
        private int hashCode;
        private int bucket;
        private int bucketPair;
        private List<String> seq;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
            itemSetLength = Utility.assertIntConfigParam(conf,  "cgs.item.set.length",  "missing item set length");
        	bucketCount = conf.getInt("cgs.bucket.count", 16);
       }

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	outVal.initialize();
        	outVal.append(0);
        	outVal.addFromArray(items, 0, itemSetLength);
        	
        	//hash self join
			seq = new ArrayList<String>();
			outVal.subTupleAsList(1, 1+itemSetLength, seq);
        	hashCode = seq.hashCode();
        	hashCode = hashCode < 0 ? -hashCode : hashCode;
        	bucket = hashCode % bucketCount;
    		for (int i = 0; i < bucketCount;  ++i) {
            	outKey.initialize();
    			bucketPair = 0;
    			if (i < bucket){
    				bucketPair = (bucket << 8) | i;
    				outKey.add(bucketPair, 1);
    				outVal.insert(1, 0);
    			} else {
    				bucketPair = (i << 8) | bucket;
    				outKey.add(bucketPair, 0);
    				outVal.insert(0, 0);
    			}
    			context.write(outKey, outVal);
    		}
        	
        }	
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class CandidateGenerationReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private String fieldDelim;
		private Text outVal  = new Text();
		private List<List<String>> sequenceList = new ArrayList<List<String>>();
		private int itemSetLength;
		private List<String> joinedSeq;
		private List<String> seq;
		
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
            itemSetLength = Utility.assertIntConfigParam(conf,  "cgs.item.set.length",  "missing item set length");
 	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<Tuple> values, Context context)
        		throws IOException, InterruptedException {
        	int bucketPair = key.getInt(0);
        	int upperBucket = bucketPair >> 8;
    		int lowerBucket = bucketPair & 0xF;
    		sequenceList.clear();
    		
    		if (lowerBucket == upperBucket) {
    			for (Tuple value : values) {
    				seq = new ArrayList<String>();
    				value.subTupleAsList(1, 1+itemSetLength, seq);
    				joinedSeq = selfJoinSequence(seq);
					if (null != joinedSeq) {
						outVal.set(Utility.join(joinedSeq, fieldDelim));
		       			context.write(NullWritable.get(),outVal);
					}
    			}
    		} else {
    			for (Tuple value : values) {
    				if (value.getInt(0) == 0) {
    					seq = new ArrayList<String>();
    					value.subTupleAsList(1, 1+itemSetLength, seq);
    					sequenceList.add(seq);
    				} else {
    					List<String> thatSeq = new ArrayList<String>();
    					value.subTupleAsList(1, 1+itemSetLength, thatSeq);
    					for (List<String> thisSeq : sequenceList) {
    						joinedSeq = joinSquences(thisSeq, thatSeq);
    						if (null != joinedSeq) {
    							outVal.set(Utility.join(joinedSeq, fieldDelim));
    			       			context.write(NullWritable.get(),outVal);
    						}
    					}
    				}
    			}
    		}
        }
        
        /**
         * @param seq
         * @return
         */
        private List<String> selfJoinSequence(List<String> seq) {
        	List<String> joinedSeq = null;
        	
        	//only if sequence contains the same token
        	String firstToken = seq.get(0);
        	boolean sameToken = true;
        	for (int i = 1; i < seq.size(); ++i) {
        		if (!seq.get(i).equals(firstToken)) {
        			sameToken = false;
        			break;
        		}
        	}
        	
        	if (sameToken) {
        		joinedSeq = new ArrayList<String>();
        		joinedSeq.addAll(seq);
        		joinedSeq.add(firstToken);
        	}
        	return joinedSeq;
        }
        
        /**
         * @param thisSeq
         * @param thatSeq
         * @return
         */
        private List<String> joinSquences(List<String> thisSeq, List<String> thatSeq) {
        	List<String> joinedSeq = null;
        	boolean matched = true;
        	
        	//slide by 1
        	for (int i = 1; i < thisSeq.size(); ++i) {
        		if (!thisSeq.get(i).equals(thatSeq.get(i - 1))) {
        			matched = false;
        			break;
        		}
        	}
        	
        	if (matched) {
        		joinedSeq = new ArrayList<String>();
        		joinedSeq.addAll(thisSeq);
        		joinedSeq.add(thatSeq.get(thatSeq.size() - 1));
        	} else  {
        		//slide by 1 other way
        		matched = true;
            	for (int i = 0; i < thisSeq.size() - 1; ++i) {
            		if (!thisSeq.get(i).equals(thatSeq.get(i + 1))) {
            			matched = false;
            			break;
            		}
            	}
        		if (matched) {
            		joinedSeq = new ArrayList<String>();
            		joinedSeq.addAll(thatSeq);
            		joinedSeq.add(thatSeq.get(thisSeq.size() - 1));
        		}
        	}
        	
        	return joinedSeq;
        }
	}	
	
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new CandidateGenerationWithSelfJoin(), args);
        System.exit(exitCode);
	}
	
}
