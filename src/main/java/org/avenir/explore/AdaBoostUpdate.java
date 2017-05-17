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
import java.util.List;

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
import org.chombo.util.Utility;

/**
 * @author pranab
 *
 */
public class AdaBoostUpdate extends Configured implements Tool {
	

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Update boost for adaboost  MR";
        job.setJobName(jobName);
        
        job.setJarByClass(AdaBoostUpdate.class);

        FileInputFormat.addInputPaths(job, args[0]);
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration());
        
        job.setMapperClass(AdaBoostUpdate.UpdateMapper.class);
 
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
	public static class UpdateMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
		private Text outVal = new Text();
        private String fieldDelimRegex;
        private String[] items;
        private int predClassAttrOrd;
        private int actualClassAttrOrd;
        private int boostAttrOrd;
        private double error;
        private double alpha;
        private double boost;
        private int outputPrecision;
        private double initialWeight;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	predClassAttrOrd = Utility.assertIntConfigParam(config, "abu.pred.class.attr.ord", 
        			"missing pedicted class attribute ordinal");
        	actualClassAttrOrd = Utility.assertIntConfigParam(config, "abu.actual.class.attr.ord", 
        			"missing actual class attribute ordinal");
        	boostAttrOrd = Utility.assertIntConfigParam(config, "abu.boost.attr.ord", 
        			"missing boost ordinal");
        	String errorFilePath = Utility.assertStringConfigParam(config, "abu.error.file.path", 
        			"missing error file path");
        	
        	//error output file
        	List<String> lines = Utility.getFileLines(errorFilePath);
        	String[] items = lines.get(0).split("=");
        	error = Double.parseDouble(items[1]);
        	alpha = 0.5 * Math.log((1.0 - error) / error);
        	
        	outputPrecision = config.getInt("abe.output.precision", 6);
        	initialWeight = Utility.assertDoubleConfigParam(config, "abu.intial.weight", 
        			"missing adabost intial weight");
        }	

	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
	    protected void map(LongWritable key, Text value, Context context)
	    	throws IOException, InterruptedException {
        	items  = value.toString().split(fieldDelimRegex, -1);
        	if (error < 0.5) {
        		//update current weight
	        	boost = Double.parseDouble(items[boostAttrOrd]);
	        	if (!items[actualClassAttrOrd].equals(items[predClassAttrOrd])) {
	        		//incorrect prediction
	        		boost *= Math.exp(alpha);
	        	} else {
	        		//correct prediction
	        		boost *= Math.exp(-alpha);
	        	}
        	} else {
        		//reset to initial weight
        		boost = initialWeight;
        	}
        	items[boostAttrOrd] = BasicUtils.formatDouble(boost, outputPrecision);
        	outVal.set(BasicUtils.join(items, fieldDelimRegex));
            context.write(NullWritable.get(), outVal);
        }
	}
	
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new AdaBoostUpdate(), args);
        System.exit(exitCode);
	}

}
