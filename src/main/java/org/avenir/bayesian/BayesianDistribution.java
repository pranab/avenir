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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Triplet;
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
        job.setMapOutputValueClass(Tuple.class);

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
	public static class DistributionMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private String[] items;
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
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
        private long val;
        private long valSq;
        private final int ONE = 1;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	tabularInput = config.getBoolean("bad.tabular.input", true);
        	if (tabularInput) {
        		//tabular input
	        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "bad.feature.schema.file.path");
	            ObjectMapper mapper = new ObjectMapper();
	            schema = mapper.readValue(fs, FeatureSchema.class);
	            
	            //class attribute field
	            classAttrField = schema.findClassAttrField();
	        	fields = schema.getFields();
        	} else {
        		//text input
                analyzer = new StandardAnalyzer(Version.LUCENE_44);
                featureAttrOrdinal = 1;
                outVal.initialize();
				outVal.add(ONE);
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
	        			boolean binned = true;
	        			featureAttrVal = items[field.getOrdinal()];
	        			featureAttrOrdinal = field.getOrdinal();
	        			if  (field.isCategorical()) {
	        				featureAttrBin= featureAttrVal;
	        			} else {
	        				if (field.isBucketWidthDefined()) {
	        					bin = Integer.parseInt(featureAttrVal) / field.getBucketWidth();
	        					featureAttrBin = "" + bin;
	        				} else {
	        					binned = false;
	        					val = Integer.parseInt(featureAttrVal);
	        					valSq = val * val;
	        				}
	        			}
	        			
	        			outKey.initialize();
	        			outVal.initialize();
	        			if (binned) {
	        				//1.cjass attribute vale 2.feature attribute ordinal 3. feature attribute bin
	        				outKey.add(classAttrVal, featureAttrOrdinal, featureAttrBin);
	        				outVal.add(ONE);
	        			} else {
	        				//1.cjass attribute vale 2.feature attribute ordinal 
	        				outKey.add(classAttrVal, featureAttrOrdinal);
	        				outVal.add(ONE, val, valSq);
	        			}
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
	public static class DistributionReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private Text outVal = new Text();
		private String fieldDelim;
		private FeatureSchema schema;
        private List<FeatureField> fields;
		private int count;
		private StringBuilder stBld = new  StringBuilder();
		private long valSum;
		private long valSqSum;
		private long featurePosteriorMean;
		private long featurePosteriorStdDev;
		private Map<Integer, Triplet<Integer, Long, Long>> featurePriorDistr = 
				new HashMap<Integer, Triplet<Integer, Long, Long>>();
		private Integer featureOrd;
		private boolean tabularInput;
		private boolean binned;
		private String classAttrValue;
		private FeatureField field;
		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelim = context.getConfiguration().get("field.delim.out", ",");
        	tabularInput = config.getBoolean("bad.tabular.input", true);
    		
        	//tabular input
        	if (tabularInput) {
        		InputStream fs = Utility.getFileStream(context.getConfiguration(), "bad.feature.schema.file.path");
        		ObjectMapper mapper = new ObjectMapper();
        		schema = mapper.readValue(fs, FeatureSchema.class);
        	}
		}

		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void cleanup(Context context) throws IOException, InterruptedException {
			//emit feature prior probability parameters for numerical continuous variables
			for (int featureOrd : featurePriorDistr.keySet()) {
				context.getCounter("Distribution Data", "Feature prior cont ").increment(1);
    			Triplet<Integer, Long, Long> distr = featurePriorDistr.get(featureOrd);
    			count = distr.getLeft();
    			valSum = distr.getCenter();
    			valSqSum = distr.getRight();
    			long mean = valSum / count;
    			double temp = valSqSum - count * mean  * mean;
    			long stdDev = (long)(Math.sqrt(temp / (count -1)));
    			
	    		stBld.delete(0, stBld.length());
	    		stBld.append(fieldDelim).append(featureOrd).append(fieldDelim).append(fieldDelim).append(mean).
	    			append(fieldDelim).append(stdDev);
	    		outVal.set(stBld.toString());
				context.write(NullWritable.get(),outVal);
			}
		}	
		
    	/* (non-Javadoc)
    	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
    	 */
    	protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
            	throws IOException, InterruptedException {
    		count = 0;
    		classAttrValue = key.getString(0);
			featureOrd = key.getInt(1);
			field = schema.findFieldByOrdinal(featureOrd);
			binned = !tabularInput ||  field.isCategorical() || field.isBucketWidthDefined();
    		if (!binned){
    			valSum = valSqSum = 0;
    		}
    		for (Tuple val : values) {
    			count += val.getInt(0);
    			if (!binned) {
    				valSum += val.getLong(1);
    				valSqSum += val.getLong(2);
    			}
    		}
    		
    		if (!binned) {
    			featurePosteriorMean = valSum / count;
    			double temp = valSqSum - count * featurePosteriorMean  * featurePosteriorMean;
    			featurePosteriorStdDev = (long)(Math.sqrt(temp / (count -1)));
    			
				//collect feature prior values across all class attribute values
    			Triplet<Integer, Long, Long> distr = featurePriorDistr.get(featureOrd);
    			if (null == distr) {
    				distr = new Triplet<Integer, Long, Long>(count, valSum, valSqSum);
    				featurePriorDistr.put(featureOrd, distr);
    			} else {
    				distr.setLeft(distr.getLeft() + count);
    				distr.setCenter(distr.getCenter() + valSum);
    				distr.setRight(distr.getRight() + valSqSum);
    			}
    		}
    		
    		//emit feature posterior
    		stBld.delete(0, stBld.length());
    		if (binned) {
				context.getCounter("Distribution Data", "Feature posterior binned ").increment(1);
    			stBld.append(key.toString()).append(fieldDelim).append(count);
    		} else {
				context.getCounter("Distribution Data", "Feature posterior cont ").increment(1);
    			stBld.append(key.toString()).append(fieldDelim).append(fieldDelim).append(featurePosteriorMean).
    			append(fieldDelim).append(featurePosteriorStdDev);
    		}
    		outVal.set(stBld.toString());
			context.write(NullWritable.get(),outVal);
			
			//emit class prior
			context.getCounter("Distribution Data", "Class prior").increment(1);
    		stBld.delete(0, stBld.length());
    		stBld.append(key.getString(0)).append(fieldDelim).append(fieldDelim).append(fieldDelim).append(count);
    		outVal.set(stBld.toString());
			context.write(NullWritable.get(),outVal);
			
			//feature prior
    		if (binned) {
				context.getCounter("Distribution Data", "Feature prior binned ").increment(1);
	    		stBld.delete(0, stBld.length());
	    		stBld.append(fieldDelim).append(key.getInt(1)).append(fieldDelim).append(key.getString(2)).
	    			append(fieldDelim).append(count);
	    		outVal.set(stBld.toString());
				context.write(NullWritable.get(),outVal);
    		} 
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
