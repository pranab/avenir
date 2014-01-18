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

package org.avenir.regress;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.chombo.mr.FeatureSchema;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

public class LogisticRegressionJob  extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Logistic regression";
        job.setJobName(jobName);
        job.setJarByClass(LogisticRegressionJob.class);
        Utility.setConfiguration(job.getConfiguration(), "avenir");

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        job.setMapperClass(LogisticRegressionJob.RegressionMapper.class);
       // job.setReducerClass(LogisticRegressionJob.PartitionGeneratorReducer.class);
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setNumReduceTasks(job.getConfiguration().getInt("num.reducer", 1));

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class RegressionMapper extends Mapper<LongWritable, Text, Text, Tuple> {
		private String fieldDelimRegex;
		private String[] items;
		private Text outVal  = new Text();
        private FeatureSchema schema;
        private int[] featureValues;
        private int[] featureOrdinals;
        private int classOrdinal;
        private String classVal;
        private int  iterCount;
        private double[] coefficients;
        private static final Logger LOG = Logger.getLogger(RegressionMapper.class);
       
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            
            //regression coefficients
            List<String[]>   lines = Utility.parseFileLines(conf, "coeff.file.path", fieldDelimRegex);
            iterCount = lines.size();
            String[] items   = lines.get(lines.size() - 1);
            coefficients = new double[items.length];
            for (int i = 0; i < items.length; ++i) {
            	coefficients[i] = Double.parseDouble(items[i]);
            }
        }       
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void cleanup(Context context)  throws IOException, InterruptedException {
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            if (null == featureValues) {
            	featureOrdinals = schema.getFeatureFieldOrdinals();
            	featureValues = new int[featureOrdinals.length + 1];
            	featureValues[0] = 1;
            	classOrdinal = schema.findClassAttrField().getOrdinal();
            }
            
            for (int i=0;  i <  featureOrdinals.length; ++i) {
            	featureValues[i] = Integer.parseInt(items[featureOrdinals[i]]);
            }
            classVal = items[classOrdinal];
            
            
        }       
	}	
}
