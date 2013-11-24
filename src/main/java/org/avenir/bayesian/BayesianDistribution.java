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

package org.avenir.bayesian;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

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
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.chombo.mr.FeatureField;
import org.chombo.mr.FeatureSchema;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * Calculates all distributions for  bayesian classifier
 * @author pranab
 *
 */
public class BayesianDistribution extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "prior and posterior distribution   MR";
        job.setJobName(jobName);
        
        job.setJarByClass(BayesianDistribution.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        
        job.setMapperClass(BayesianDistribution.DistributionMapper.class);
        job.setReducerClass(BayesianDistribution.DistributionReducer.class);

        job.setMapOutputKeyClass(Tuple.class);
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
	public static class DistributionMapper extends Mapper<LongWritable, Text, Tuple, IntWritable> {
		private String[] items;
		private Tuple outKey = new Tuple();
		private IntWritable outVal = new IntWritable(1);
        private String fieldDelimRegex;
        private FeatureSchema schema;
        private List<FeatureField> fields;
        private FeatureField classAttrField;
        private String classAttrVal;
        private String featureAttrVal;
        private Integer featureAttrOrdinal;
        private String featureAttrBin;
        private int bin;
        private boolean tabularInput;
        private Analyzer analyzer;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	fieldDelimRegex = context.getConfiguration().get("bs.field.delim.regex", ",");
        	tabularInput = context.getConfiguration().getBoolean("tabular.input", true);
        	if (tabularInput) {
	        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
	            ObjectMapper mapper = new ObjectMapper();
	            schema = mapper.readValue(fs, FeatureSchema.class);
	            
	            //class attribute field
	            classAttrField = schema.findClassAttrField();
	        	fields = schema.getFields();
        	} else {
                analyzer = new StandardAnalyzer(Version.LUCENE_35);
                featureAttrOrdinal = 1;
        	}
        }
 
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        	if (tabularInput) {
	            items  =  value.toString().split(fieldDelimRegex);
	            classAttrVal = items[classAttrField.getOrdinal()];
	            
	        	for (FeatureField field : fields) {
	        		if (field.isFeature()) {
	        			featureAttrVal = items[field.getOrdinal()];
	        			featureAttrOrdinal = field.getOrdinal();
	        			if  (field.isCategorical()) {
	        				featureAttrBin= featureAttrVal;
	        			} else {
	        				bin = Integer.parseInt(featureAttrVal) / field.getBucketWidth();
	        				featureAttrBin = "" + bin;
	        			}
	        			outKey.initialize();
	        			outKey.add(classAttrVal, featureAttrOrdinal, featureAttrBin);
	       	   			context.write(outKey, outVal);
	        		}
	        	}
        	}   else {
        		mapText( value,  context);
        	}
        }
        
        /**
         * @param value
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void mapText(Text value, Context context) throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            classAttrVal = items[1];
            List<String> tokens = Utility.tokenize(items[0], analyzer);
            for (String token : tokens ) {
    			outKey.initialize();
    			outKey.add(classAttrVal, featureAttrOrdinal, token);
   	   			context.write(outKey, outVal);
            }
        }
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class DistributionReducer extends Reducer<Tuple, IntWritable, NullWritable, Text> {
		private Text outVal = new Text();
		private String fieldDelim;
		private int count;
		private StringBuilder stBld = new  StringBuilder();
		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
        	fieldDelim = context.getConfiguration().get("field.delim.out", ",");
       }

    	/* (non-Javadoc)
    	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
    	 */
    	protected void reduce(Tuple key, Iterable<IntWritable> values, Context context)
            	throws IOException, InterruptedException {
    		count = 0;
    		for (IntWritable val : values) {
    			++count;
    		}
    		//feature posterior
    		stBld.delete(0, stBld.length());
    		stBld.append(key.toString()).append(fieldDelim).append(count);
    		outVal.set(stBld.toString());
			context.write(NullWritable.get(),outVal);
			
			//class prior
    		stBld.delete(0, stBld.length());
    		stBld.append(key.getString(0)).append(fieldDelim).append(fieldDelim).append(fieldDelim).append(count);
    		outVal.set(stBld.toString());
			context.write(NullWritable.get(),outVal);
			
			//feature prior
    		stBld.delete(0, stBld.length());
    		stBld.append(fieldDelim).append(key.getInt(1)).append(fieldDelim).append(key.getString(2)).append(fieldDelim).append(count);
    		outVal.set(stBld.toString());
			context.write(NullWritable.get(),outVal);
			
    	}
	}
	
    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new BayesianDistribution(), args);
        System.exit(exitCode);
    }
	
}
