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

package org.avenir.explore;

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
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.util.BasicUtils;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Calculate classification error weighted by sample boost
 * @author pranab
 *
 */
public class AdaBoostError extends Configured implements Tool {
	

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Boost weighted classification error  MR";
        job.setJobName(jobName);
        
        job.setJarByClass(AdaBoostError.class);

        FileInputFormat.addInputPaths(job, args[0]);
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration());
        
        job.setMapperClass(AdaBoostError.ErrorMapper.class);
        job.setReducerClass(AdaBoostError.ErrorReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("abe.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);
        
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class ErrorMapper extends Mapper<LongWritable, Text, Text, Tuple> {
		private Text outKey = new Text("errorStat");
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private String[] items;
        private int predClassAttrOrd;
        private int actualClassAttrOrd;
        private int boostAttrOrd;
        private double errorSum = 0;
        private int errorCount = 0;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	predClassAttrOrd = Utility.assertIntConfigParam(config, "abe.pred.class.attr.ord", 
        			"missing pedicted class attribute ordinal");
        	actualClassAttrOrd = Utility.assertIntConfigParam(config, "abe.actual.class.attr.ord", 
        			"missing actual class attribute ordinal");
        	boostAttrOrd = Utility.assertIntConfigParam(config, "abe.boost.attr.ord", 
        			"missing boost ordinal");
        }	

	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
	   	 */
	   	@Override
	   	protected void cleanup(Context context)  throws IOException, InterruptedException {
	   		outVal.initialize();
	   		outVal.add(errorCount, errorSum);
        	context.write(outKey, outVal);
	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
	    protected void map(LongWritable key, Text value, Context context)
	    	throws IOException, InterruptedException {
        	items  = value.toString().split(fieldDelimRegex, -1);
        	if (!items[actualClassAttrOrd].equals(items[predClassAttrOrd])) {
        		errorSum += Double.parseDouble(items[boostAttrOrd]);
        	}
    		++errorCount;
        }
	}
	
    /**
     * @author pranab
     *
     */
    public static class ErrorReducer extends Reducer<Text, Tuple, NullWritable, Text> {
		private Text outVal = new Text();
		private String fieldDelimOut;
		private StringBuilder stBld = new  StringBuilder();
		private int outputPrecision;
		private boolean weightNormalized;

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimOut = config.get("field.delim", ",");
        	outputPrecision = config.getInt("abe.output.precision", 6);
        	weightNormalized = config.getBoolean("abe.weight.normalized", false);
        }
    
		/* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
            double errorSum = 0;
            int errorCount = 0;
            
            for (Tuple value : values){
            	errorCount += value.getInt(0);
            	errorSum += value.getDouble(1);
            }
            double error = weightNormalized? errorSum : errorSum / errorCount;
            outVal.set("error=" + BasicUtils.formatDouble(error, outputPrecision));
            context.write(NullWritable.get(), outVal);
        }
    }
    
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new AdaBoostError(), args);
        System.exit(exitCode);
	}
    
}
