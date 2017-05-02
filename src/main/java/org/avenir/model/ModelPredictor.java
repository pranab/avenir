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


package org.avenir.model;

import java.io.IOException;

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
import org.chombo.util.FeatureSchema;
import org.chombo.util.Utility;

/**
 * @author pranab
 *
 */
public class ModelPredictor extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "model predictor   MR";
        job.setJobName(jobName);
        
        job.setJarByClass(ModelPredictor.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        
        job.setMapperClass(ModelPredictor.PredictorMapper.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class PredictorMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
		private String[] items;
		private Text outVal = new Text();
        private String fieldDelimRegex;
        private String fieldDelim;
        private FeatureSchema schema;
		private String[] classAttrValues;
		private int probThreshHold = 50;

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	fieldDelim = config.get("field.delim.out", ",");

        	//schema
            schema = Utility.getFeatureSchema(config, "mop.feature.schema.file.path");
            classAttrValues = Utility.assertStringArrayConfigParam(config, "mop.class.attr.values", Utility.configDelim, 
            		"missing class attribute values");
            double falsePosErrorCost = (double)config.getFloat("mop.false.pos.error.cost", (float)1.0);
            double falseNegErrorCost = (double)config.getFloat("mop.false.neg.error.cost", (float)1.0);
            
        }   
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
        }        
	}	
	
	
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new ModelPredictor(), args);
        System.exit(exitCode);
    }

}
