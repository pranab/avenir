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
import org.chombo.mr.FeatureField;
import org.chombo.mr.FeatureSchema;
import org.chombo.util.Pair;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * Calculates various mutual information between feature and class attributes. Can be used
 * for relevant feature sub set selection
 * @author pranab
 *
 */
public class MutualInformation extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "mutual information   MR";
        job.setJobName(jobName);
        
        job.setJarByClass(MutualInformation.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        
        job.setMapperClass(MutualInformation.DistributionMapper.class);
        job.setReducerClass(MutualInformation.DistributionReducer.class);

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
        private FeatureField classAttrField;
        private List<FeatureField> featureFields;
        private String classAttrVal;
        private String featureAttrVal;
        private Integer featureAttrOrdinal;
        private String featureAttrBin;
        private String firstFeatureAttrBin;
        private int bin;
        private static final int CLASS_DIST = 1;
        private static final int FEATURE_DIST = 2;
        private static final int FEATURE_PAIR_DIST = 3;
        private static final int FEATURE_CLASS_DIST = 4;
        private static final int FEATURE_PAIR_CLASS_DIST = 5;
        private static final int FEATURE_CLASS_COND_DIST = 6;
        private static final int FEATURE_PAIR_CLASS_COND_DIST = 7;

        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	fieldDelimRegex = context.getConfiguration().get("field.delim.regex", ",");
        	
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            
            classAttrField = schema.findClassAttrField();
            featureFields = schema.getFeatureAttrFields();
        }        
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            classAttrVal = items[classAttrField.getOrdinal()];
            
            //class
        	outKey.initialize();
        	outVal.initialize();
        	outKey.add(CLASS_DIST,classAttrField.getOrdinal());
			outVal.add(classAttrVal, 1);
	   		context.write(outKey, outVal);

            //feature and feature class
            for (FeatureField field : featureFields) {
            	outKey.initialize();
            	outVal.initialize();
            	
    			featureAttrVal = items[field.getOrdinal()];
    			setDistrValue(field);

    			//feature
    			outKey.add(FEATURE_DIST, field.getOrdinal());
    			outVal.add(featureAttrBin, 1);
   	   			context.write(outKey, outVal);
   	   			
   	   			//feature class
   	   			outKey.set(0,  FEATURE_CLASS_DIST);
   	   			outVal.initialize();
   	   			outVal.add(featureAttrBin, classAttrVal, 1);
   	   			context.write(outKey, outVal);
   	   			
   	   			//feature class conditional
   	   			outKey.set(0,  FEATURE_CLASS_COND_DIST);
   	   			outKey.add(classAttrVal);
  	   			outVal.initialize();
   	   			outVal.add(featureAttrBin,  1);
   	   			context.write(outKey, outVal);
   	   			
            }
            
            //feature pair and feature pair class conditional
            for (int i = 0; i < featureFields.size(); ++i) {
    			featureAttrVal = items[featureFields.get(i).getOrdinal()];
    			setDistrValue(featureFields.get(i));
    			firstFeatureAttrBin = featureAttrBin;
            	for (int j = i+1; j < featureFields.size(); ++j) {
                	outKey.initialize();
                	outVal.initialize();
        			featureAttrVal = items[featureFields.get(j).getOrdinal()];
        			setDistrValue(featureFields.get(j));

        			//feature pairs
        			outKey.add(FEATURE_PAIR_DIST, featureFields.get(i).getOrdinal(), 
        					featureFields.get(j).getOrdinal());
         			outVal.add(firstFeatureAttrBin, featureAttrBin, 1);
       	   			context.write(outKey, outVal);
       	   			
       	   			//feature pair class 
                	outKey.initialize();
                	outVal.initialize();
        			outKey.add(FEATURE_PAIR_CLASS_DIST, featureFields.get(i).getOrdinal(), 
        					featureFields.get(j).getOrdinal());
         			outVal.add(firstFeatureAttrBin, featureAttrBin, classAttrVal, 1);
       	   			context.write(outKey, outVal);
       	   			
       	   			
       	   			//feature pair class conditional
                	outKey.initialize();
                	outVal.initialize();
       	   			outKey.add(FEATURE_PAIR_CLASS_COND_DIST, featureFields.get(i).getOrdinal(), 
        					featureFields.get(j).getOrdinal(), classAttrVal);
         			outVal.add(firstFeatureAttrBin, featureAttrBin, 1);
       	   			context.write(outKey, outVal);
            	}
            }

        }     
        
    	/**
    	 * @param field
    	 */
    	private void setDistrValue(FeatureField field) {
    		if  (field.isCategorical()) {
    			featureAttrBin= featureAttrVal;
    		} else {
    			bin = Integer.parseInt(featureAttrVal) / field.getBucketWidth();
    			featureAttrBin = "" + bin;
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
		private int distrType;
		private String attrValue;
		private String secondAttrValue;
		private Pair<String, String> attrValuePair;
		private Tuple attrPairClassValue;
		private int attrCount;
		private Integer curCount;
        private FeatureSchema schema;
		private Map<String, Integer> classDistr = new HashMap<String, Integer>();
		private Map<Integer, Map<String, Integer>> allFeatureDistr = new HashMap<Integer, Map<String, Integer>>();
		private Map<Pair<Integer, Integer>, Map<Pair<String, String>, Integer>> allFeaturePairDistr = 
				new HashMap<Pair<Integer, Integer>, Map<Pair<String, String>, Integer>>();
		private Map<Integer, Map<Pair<String, String>, Integer>> allFeatureClassDistr = 
				new HashMap<Integer, Map<Pair<String, String>, Integer>>();
		private Map<Pair<Integer, Integer>, Map<Tuple, Integer>> allFeaturePairClassDistr = 
				new HashMap<Pair<Integer, Integer>, Map<Tuple, Integer>>();
		private Map<Pair<Integer,String>, Map<String, Integer>> allFeatureClassCondDistr = 
				new HashMap<Pair<Integer,String>, Map<String, Integer>>();
		private Map<Tuple, Map<Pair<String, String>, Integer>> allFeaturePairClassCondDistr = 
				new HashMap<Tuple, Map<Pair<String, String>, Integer>>();
		private StringBuilder stBld = new StringBuilder();
		private int totalCount;
		private boolean outputMutualInfo;
        private static final int CLASS_DIST = 1;
        private static final int FEATURE_DIST = 2;
        private static final int FEATURE_PAIR_DIST = 3;
        private static final int FEATURE_CLASS_DIST = 4;
        private static final int FEATURE_PAIR_CLASS_DIST = 5;
        private static final int FEATURE_CLASS_COND_DIST = 6;
        private static final int FEATURE_PAIR_CLASS_COND_DIST = 7;
		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
	    	fieldDelim = conf.get("field.delim.out", ",");

	    	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            outputMutualInfo = conf.getBoolean("output.mutual.info", true);
		}
		
	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	@Override
	   	protected void cleanup(Context context)  throws IOException, InterruptedException {
	   		//emit distributions
	   		outputDistr(context);
	   		
	   		//emit mutual information
	   		if (outputMutualInfo) {
	   			outputMutualInfo(context);
	   		}
	   	}
	   	
	   	/**
	   	 * @param context
	   	 * @throws IOException
	   	 * @throws InterruptedException
	   	 */
	   	private void outputDistr(Context context) throws IOException, InterruptedException {
	   		//class 
	   		outVal.set("distribution:class");
	   		context.write(NullWritable.get(), outVal);
	   		totalCount = 0;
	   		for (String classVal :  classDistr.keySet()) {
	   			totalCount += classDistr.get(classVal);
	   		}
	   		for (String classVal :  classDistr.keySet()) {
	   			stBld.delete(0, stBld.length());
	   			stBld.append(classVal).append(fieldDelim).append(((double)classDistr.get(classVal)) / totalCount);;
		   		outVal.set(stBld.toString());
		   		context.write(NullWritable.get(), outVal);
	   		}
	   		
	   		//feature
	   		outVal.set("distribution:feature");
	   		context.write(NullWritable.get(), outVal);
	   		for (int featureOrd : allFeatureDistr.keySet()) {
	   			Map<String, Integer> featureDistr = allFeatureDistr.get(featureOrd);
		   		for (String featureVal :  featureDistr.keySet()) {
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrd).append(fieldDelim).append(featureVal).append(fieldDelim).
		   				append(((double)featureDistr.get(featureVal)) / totalCount);;
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
		   		}
	   		}
	   		
	   		//feature pair
	   		outVal.set("distribution:featurePair");
	   		context.write(NullWritable.get(), outVal);
	   		for (Pair<Integer, Integer> featureOrdinals : allFeaturePairDistr.keySet()) {
	   			Map<Pair<String, String>, Integer> featurePairDistr = allFeaturePairDistr.get(featureOrdinals);
	   			for (Pair<String, String> values : featurePairDistr.keySet()) {
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrdinals.getLeft()).append(fieldDelim).append(featureOrdinals.getRight()).
		   				append(fieldDelim).append(values.getLeft()).append(fieldDelim).append(values.getRight()).
		   				append(fieldDelim).append(((double)featurePairDistr.get(values)) / totalCount);
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
	   			}
	   		}
	   		
	   		//feature class
	   		outVal.set("distribution:featureClass");
	   		context.write(NullWritable.get(), outVal);
	   		for (Integer featureOrd : allFeatureClassDistr.keySet()) {
	   			Map<Pair<String, String>, Integer> featureClassDistr = allFeatureClassDistr.get(featureOrd);
	   			for (Pair<String, String> values : featureClassDistr.keySet()) {
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrd).append(fieldDelim).append(values.getLeft()).append(fieldDelim).
		   				append(values.getRight()). append(fieldDelim).append(((double)featureClassDistr.get(values)) / 
		   				totalCount);
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
	   			}
	   		}	 
	   		
	   		//feature pair class
	   		outVal.set("distribution:featurePairClass");
	   		context.write(NullWritable.get(), outVal);
	   		for (Pair<Integer, Integer> featureOrdinals : allFeaturePairClassDistr.keySet()) {
	   			Map<Tuple, Integer> featurePairClassDistr = allFeaturePairClassDistr.get(featureOrdinals);
	   			for (Tuple values : featurePairClassDistr.keySet()) {
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrdinals.getLeft()).append(fieldDelim).append(featureOrdinals.getRight()).
		   				append(fieldDelim).append(values.getString(0)).append(fieldDelim).append(values.getString(1)).
		   				append(fieldDelim).append(values.getString(2)).append(fieldDelim).
		   				append(((double)featurePairClassDistr.get(values)) / totalCount);
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
	   			}
	   		}
	   		
	   		
	   		//feature class conditional
	   		outVal.set("distribution:featureClassConditional");
	   		context.write(NullWritable.get(), outVal);
	   		for (Pair<Integer, String> featureOrdinalClassVal : allFeatureClassCondDistr.keySet()) {
	   			Map<String, Integer> featureDistr = allFeatureClassCondDistr.get(featureOrdinalClassVal);
		   		for (String featureVal :  featureDistr.keySet()) {
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrdinalClassVal.getLeft()).append(fieldDelim).
		   				append(featureOrdinalClassVal.getRight()).append(fieldDelim).append(featureVal).append(fieldDelim).
		   				append(((double)featureDistr.get(featureVal)) / totalCount);;
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
		   		}
	   		}
	  
	   		//feature pair class conditional
	   		outVal.set("distribution:featurePairClassConditional");
	   		context.write(NullWritable.get(), outVal);
	   		for (Tuple featureOrdinalsClassVal : allFeaturePairClassCondDistr.keySet()) {
	   			Map<Pair<String, String>, Integer> featurePairDistr = 
	   					allFeaturePairClassCondDistr.get(featureOrdinalsClassVal);
	   			for (Pair<String, String> values : featurePairDistr.keySet()) {
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrdinalsClassVal.getInt(0)).append(fieldDelim).
		   				append(featureOrdinalsClassVal.getInt(1)).append(fieldDelim).
		   				append(featureOrdinalsClassVal.getString(2)).append(fieldDelim).
		   				append(values.getLeft()).append(fieldDelim).append(values.getRight()).
		   				append(fieldDelim).append(((double)featurePairDistr.get(values)) / totalCount);
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
	   			}
	   		}
	   		
	   	}
	   	
	   	/**
	   	 * @param context
	   	 * @throws IOException
	   	 * @throws InterruptedException
	   	 */
	   	private void outputMutualInfo(Context context) throws IOException, InterruptedException {
   			double sum = 0;
   			double jointProb;
   			double featureProb;
   			double featureProbFirst;
   			double featureProbSecond;
   			double classProb;
   			Integer count;
   			int numSamples;
   			Tuple featurePairClass = new Tuple();
   			Tuple featureOrdinalsClassCond = new Tuple();

   			//feature class
	   		outVal.set("mutualInformation:feature");
	   		context.write(NullWritable.get(), outVal);
   			int[] featureOrdinals = schema.getFeatureFieldOrdinals();
	   		for (int featureOrd : allFeatureDistr.keySet()) {
	   			sum = 0;
	   			Map<String, Integer> featureDistr = allFeatureDistr.get(featureOrd);
	   			numSamples = 0;
		   		for (String featureVal :  featureDistr.keySet()) {
		   			featureProb = ((double)featureDistr.get(featureVal)) / totalCount;
			   		for (String classVal :  classDistr.keySet()) {
			   			classProb = ((double)classDistr.get(classVal)) / totalCount;
			   			Pair<String, String> values = new Pair(featureVal, classVal);
			   			count = allFeatureClassDistr.get(featureOrd).get(values);
			   			if (null != count) {
			   				jointProb = ((double)count) / totalCount;
			   				sum += jointProb * Math.log(jointProb / (featureProb * classProb));
			   				++numSamples;
			   			}
			   		}
		   		}
		   		double featureClassMI = sum;
	   			stBld.delete(0, stBld.length());
	   			stBld.append(featureOrd).append(fieldDelim).append(featureClassMI);
		   		outVal.set(stBld.toString());
		   		context.write(NullWritable.get(), outVal);
	   		}
	   		
	   		//feature pair
	   		outVal.set("mutualInformation:featurePair");
	   		context.write(NullWritable.get(), outVal);
	   		for (int i = 0; i < featureOrdinals.length; ++i) {
	   			Map<String, Integer> featureDistrFirst = allFeatureDistr.get(featureOrdinals[i]);
	   			for (int j = i+1; j < featureOrdinals.length; ++j) {
	   				Pair<Integer, Integer> featureOrdinalPair = 
	   						new Pair<Integer, Integer>(featureOrdinals[i], featureOrdinals[j]);
	   				Map<Pair<String, String>, Integer> featurePairDistr = allFeaturePairDistr.get(featureOrdinalPair);
		   			Map<String, Integer> featureDistrSecond = allFeatureDistr.get(featureOrdinals[j]);
		   			sum = 0;
		   			numSamples = 0;
			   		for (String featureValFirst :  featureDistrFirst.keySet()) {
			   			featureProbFirst = ((double)featureDistrFirst.get(featureValFirst)) / totalCount;
				   		for (String featureValSecond :  featureDistrSecond.keySet()) {
				   			featureProbSecond = ((double)featureDistrSecond.get(featureValSecond)) / 
				   					totalCount;
				   			Pair<String, String> values = 
				   					new Pair<String, String>(featureValFirst, featureValSecond);
				   			count = featurePairDistr.get(values);
				   			if (null != count) {
				   				jointProb = ((double)count) / totalCount;
				   				sum += jointProb * Math.log(jointProb / (featureProbFirst * featureProbSecond));
				   				++numSamples;
				   			}
				   		}
			   		}
			   		double featurePairMI = sum;
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrdinals[i]).append(fieldDelim).append(featureOrdinals[j]).
		   				append(fieldDelim).append(featurePairMI);
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
	   			}
	   		}
	   		
	   		//feature pair class
	   		outVal.set("mutualInformation:featurePairClass");
	   		context.write(NullWritable.get(), outVal);
	   		for (int i = 0; i < featureOrdinals.length; ++i) {
	   			Map<String, Integer> featureDistrFirst = allFeatureDistr.get(featureOrdinals[i]);
	   			for (int j = i+1; j < featureOrdinals.length; ++j) {
		   			Map<String, Integer> featureDistrSecond = allFeatureDistr.get(featureOrdinals[j]);
	   				Pair<Integer, Integer> featureOrdinalPair = 
	   						new Pair<Integer, Integer>(featureOrdinals[i], featureOrdinals[j]);
	   				Map<Tuple, Integer> featurePairClassDistr = allFeaturePairClassDistr.get(featureOrdinalPair);
		   			sum = 0;
		   			numSamples = 0;
			   		for (String featureValFirst :  featureDistrFirst.keySet()) {
			   			featureProbFirst = ((double)featureDistrFirst.get(featureValFirst)) / totalCount;
				   		for (String featureValSecond :  featureDistrSecond.keySet()) {
				   			featureProbSecond = ((double)featureDistrSecond.get(featureValSecond)) / 
				   					totalCount;
			   				for (String classVal : classDistr.keySet()) {
			   					classProb = ((double)classDistr.get(classVal)) / totalCount;
			   					featurePairClass.initialize();
			   					featurePairClass.add(featureValFirst, featureValSecond, classVal);
			   					count = featurePairClassDistr.get(featurePairClass);
			   					if (null != count) {
					   				jointProb = ((double)count) / totalCount;
					   				sum += jointProb * Math.log(jointProb / 
					   						(featureProbFirst * featureProbSecond * classProb));
					   				++numSamples;
			   					}
			   				}
				   		}
			   		}
			   		double featurePairClassMI = sum;
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrdinals[i]).append(fieldDelim).append(featureOrdinals[j]).
		   				append(fieldDelim).append(featurePairClassMI);
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
	   			}
	   		}
	   		
	   		//feature pair class conditional
	   		outVal.set("mutualInformation:featurePairClassConditional");
	   		context.write(NullWritable.get(), outVal);
	   		for (int i = 0; i < featureOrdinals.length; ++i) {
	   			for (int j = i+1; j < featureOrdinals.length; ++j) {
	   		   		double featurePairClassCondMI = 0;
	   				for (String classVal :  classDistr.keySet()) {
	   					classProb = ((double)classDistr.get(classVal)) / totalCount;
	   					Pair<Integer,String> firstFeatureClassCond = 
		   					new Pair<Integer,String>(featureOrdinals[i], classVal); 
	   					Map<String, Integer> featureDistrFirst = allFeatureClassCondDistr.get(firstFeatureClassCond);
			   			Pair<Integer,String> secondFeatureClassCond = 
			   					new Pair<Integer,String>(featureOrdinals[i], classVal); 
			   			Map<String, Integer> featureDistrSecond = allFeatureClassCondDistr.get(secondFeatureClassCond);

			   			//joint distr
			   			featureOrdinalsClassCond.initialize();
			   			featureOrdinalsClassCond.add(featureOrdinals[i], featureOrdinals[j], classVal);
			   			Map<Pair<String, String>, Integer> featurePairDistr = 
			   					allFeaturePairClassCondDistr.get(featureOrdinalsClassCond);
			   			sum = 0;
			   			numSamples = 0;
				   		for (String featureValFirst :  featureDistrFirst.keySet()) {
				   			featureProbFirst = ((double)featureDistrFirst.get(featureValFirst)) / totalCount;
					   		for (String featureValSecond :  featureDistrSecond.keySet()) {
					   			featureProbSecond = ((double)featureDistrSecond.get(featureValSecond)) / 
					   					totalCount;
					   			Pair<String, String> values = 
					   					new Pair<String, String>(featureValFirst, featureValSecond);
					   			count = featurePairDistr.get(values);
					   			if (null != count) {
					   				jointProb = ((double)count) / totalCount;
					   				sum += jointProb * Math.log(jointProb / 
					   						(featureProbFirst * featureProbSecond));
					   				++numSamples;
					   			}
					   		}			 
				   		}
				   		featurePairClassCondMI += sum * classProb;
	   				}
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrdinals[i]).append(fieldDelim).append(featureOrdinals[j]).
		   				append(fieldDelim).append(featurePairClassCondMI);
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
	   			} 
	   		} 
	   		
	   	}
	   	
	   	
	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
            	throws IOException, InterruptedException {
	   		distrType = key.getInt(0);
	   		if (distrType == CLASS_DIST) {
	   			populateDistrMap(values, classDistr);
	   		} else if (distrType == FEATURE_DIST) {
	   			int featureOrdinal = key.getInt(1);
	   			Map<String, Integer> featureDistr = allFeatureDistr.get(featureOrdinal);
	   			if (null == featureDistr) {
	   				featureDistr = new HashMap<String, Integer>();
	   				allFeatureDistr.put(featureOrdinal, featureDistr);
	   			}
	   			populateDistrMap(values, featureDistr);
	   		} else if (distrType == FEATURE_PAIR_DIST) {
	   			Pair<Integer, Integer> featureOrdinals = new Pair<Integer, Integer>(key.getInt(1), key.getInt(2));
	   			Map<Pair<String, String>, Integer> featurePairDistr = allFeaturePairDistr.get(featureOrdinals);
	   			if (null == featurePairDistr) {
	   				featurePairDistr = new HashMap<Pair<String, String>, Integer>();
	   				allFeaturePairDistr.put(featureOrdinals, featurePairDistr);
	   			}
	   			populateJointDistrMap(values, featurePairDistr);
	   		} else if (distrType == FEATURE_CLASS_DIST) {
	   			int featureOrdinal = key.getInt(1);
	   			Map<Pair<String, String>, Integer> featureClassDistr = allFeatureClassDistr.get(featureOrdinal);
	   			if (null == featureClassDistr) {
	   				featureClassDistr = new HashMap<Pair<String, String>, Integer>();
	   				allFeatureClassDistr.put(featureOrdinal, featureClassDistr);
	   			}
	   			populateJointDistrMap(values, featureClassDistr);
	   		}else if (distrType == FEATURE_PAIR_CLASS_DIST) {
	   			Pair<Integer, Integer> featureOrdinals = new Pair<Integer, Integer>(key.getInt(1), key.getInt(2));
	   			Map<Tuple, Integer> featurePairClassDistr = allFeaturePairClassDistr.get(featureOrdinals);
	   			if (null == featurePairClassDistr) {
	   				featurePairClassDistr = new HashMap<Tuple, Integer>();
	   				allFeaturePairClassDistr.put(featureOrdinals, featurePairClassDistr);
	   			}
	   			populateJointClassDistrMap(values, featurePairClassDistr);
	   		} else if (distrType == FEATURE_CLASS_COND_DIST) {
	   			Pair<Integer, String> featureOrdinalClassVal = 
	   					new Pair<Integer, String>(key.getInt(1), key.getString(2));
	   			Map<String, Integer> featureDistr = allFeatureClassCondDistr.get(featureOrdinalClassVal);
	   			if (null == featureDistr) {
	   				featureDistr = new HashMap<String, Integer>();
	   				allFeatureClassCondDistr.put(featureOrdinalClassVal, featureDistr);
	   			}
	   			populateDistrMap(values, featureDistr);
	   		} else if (distrType == FEATURE_PAIR_CLASS_COND_DIST) {
	   			Tuple featureOrdinalsClassVal = key.subTuple(1, 4);
	   			Map<Pair<String, String>, Integer> featurePairDistr = 
	   					allFeaturePairClassCondDistr.get(featureOrdinalsClassVal);
	   			if (null == featurePairDistr) {
	   				featurePairDistr = new HashMap<Pair<String, String>, Integer>();
	   				allFeaturePairClassCondDistr.put(featureOrdinalsClassVal, featurePairDistr);
	   			}
	   			populateJointDistrMap(values, featurePairDistr);
	   		}
	   		
	   	}
	   	
	   	/**
	   	 * @param values
	   	 * @param distr
	   	 */
	   	private void populateDistrMap(Iterable<Tuple> values, Map<String, Integer> distr) {
   			for (Tuple value : values) {
   				attrValue = value.getString(0);
  				attrCount = value.getInt(1);
  				curCount = distr.get(attrValue);
   				if (null == curCount) {
   					distr.put(attrValue, attrCount);
   				} else {
   					distr.put(attrValue, curCount + attrCount);
   				}
   			}
	   	}

	   	/**
	   	 * @param values
	   	 * @param distr
	   	 */
	   	private void populateJointDistrMap(Iterable<Tuple> values, Map<Pair<String, String>, Integer> distr) {
   			for (Tuple value : values) {
   				attrValue = value.getString(0);
   				secondAttrValue = value.getString(1);
   				attrCount = value.getInt(2);
   				attrValuePair = new Pair<String, String>(attrValue, secondAttrValue);
  				   				
  				curCount = distr.get(attrValuePair);
   				if (null == curCount) {
   					distr.put(attrValuePair, attrCount);
   				} else {
   					distr.put(attrValuePair, curCount + attrCount);
   				}
   			}	   		
	   	}	   	

	   	/**
	   	 * @param values
	   	 * @param distr
	   	 */
	   	private void populateJointClassDistrMap(Iterable<Tuple> values, Map<Tuple, Integer> distr) {
   			for (Tuple value : values) {
   				attrCount = value.getInt(3);
   				attrPairClassValue = value.subTuple(0, 3);
  				   				
  				curCount = distr.get(attrPairClassValue);
   				if (null == curCount) {
   					distr.put(attrPairClassValue, attrCount);
   				} else {
   					distr.put(attrPairClassValue, curCount + attrCount);
   				}
   			}	   		
	   	}	   	
	
	}
	
    /**
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new MutualInformation(), args);
        System.exit(exitCode);
    }
	
}
