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

package org.avenir.knn;

import java.io.IOException;

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
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.util.SecondarySort;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Joins training vector feature class conditional probability with training vector nearest 
 * neighbours
 * @author pranab
 *
 */
public class FeatureCondProbJoiner extends Configured implements Tool {
    private static final int GR_PROBABILITY =0;
    private static final int GR_NEIGHBOUR = 1;

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Training vector feature cond probability joiner  MR";
        job.setJobName(jobName);
        
        job.setJarByClass(FeatureCondProbJoiner.class);
        
        FileInputFormat.addInputPaths(job, args[0]);
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        job.setMapperClass(FeatureCondProbJoiner.JoinerMapper.class);
        job.setReducerClass(FeatureCondProbJoiner.JoinerReducer.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        
        job.setGroupingComparatorClass(SecondarySort.TuplePairGroupComprator.class);
        job.setPartitionerClass(SecondarySort.TuplePairPartitioner.class);

        Utility.setConfiguration(job.getConfiguration());

        job.setNumReduceTasks(job.getConfiguration().getInt("num.reducer", 1));
        
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class JoinerMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private boolean isFeatureCondProbSplit;
        private String[] items;

		/* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
            fieldDelimRegex = context.getConfiguration().get("field.delim.regex", ",");
            String splitPrefix = context.getConfiguration().get("fcb.feature.cond.prob.split.prefix", "condProb");
    		isFeatureCondProbSplit = ((FileSplit)context.getInputSplit()).getPath().getName().startsWith(splitPrefix);
        }
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
        	outKey.initialize();
        	outVal.initialize();
        	
        	if (isFeatureCondProbSplit) {
        		//training itemID
        		outKey.add(items[0], GR_PROBABILITY);
        		
        		//all class conditional probabilities ending with class value and skip feature prior probability
        		for (int i = 2; i < items.length; ++i) {
        			outVal.add(items[i]);
        		}
        	} else {
        		//nearest neighbor split
           		outKey.add(items[0], GR_NEIGHBOUR);
           		
           		//test vector neighbor itemdID, distance, class
        		outVal.add(items[1], items[2],items[4]);
        	}
        	context.write(outKey, outVal);
        }       
	}
	
    /**
     * @author pranab
     *
     */
     public static class JoinerReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private Text outVal = new Text();
 		private StringBuilder stBld =  new StringBuilder();;
 		private  String fieldDelim;
 		private String trainingClassValProb;
 		private String trainITemID;
 		private String classVal;
 		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelim = config.get("field.delim.out", ",");
       }
		
    	/* (non-Javadoc)
    	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
    	 */
    	protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
    		boolean first = true;
    		trainITemID = key.getString(0);
    		for (Tuple val : values) {
    			if (first) {
    				//class value and corresponding posterior probability for training vector
            		String classVal = val.getString(val.getSize()-1);
            		for (int i = 0; i < val.getSize()-1; i+=2) {
            			if (val.getString(i).equals(classVal)) {
            				trainingClassValProb = classVal + fieldDelim + val.getString(i+1);
            				break;
            			}
            		}
    				first = false;
    			} else {
    				//0.test ItemID, 1.test Item class value, 2.trainingItemID, 3.distance, 4.traingItem class value, 
    				//5.trainingItem feature posterior probability
    		   		stBld.delete(0, stBld.length());
    		   		stBld.append(val.getString(0)).append(fieldDelim).append(val.getString(2)).append(fieldDelim).append(trainITemID).
    		   		append(fieldDelim).append(val.getString(1)).append(fieldDelim).append(trainingClassValProb);
    		   		outVal.set(stBld.toString());
    				context.write(NullWritable.get(), outVal);
    			}
    		}		
    	}
    }	
     
     public static void main(String[] args) throws Exception {
         int exitCode = ToolRunner.run(new FeatureCondProbJoiner(), args);
         System.exit(exitCode);
     }
     
}
