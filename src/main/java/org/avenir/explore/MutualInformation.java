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
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.explore.MutualInformationScore.FeatureMutualInfo;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Pair;
import org.chombo.util.TextTuple;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * Calculates various mutual information between feature and class attributes. Can be used
 * for relevant feature sub set selection. Evaluates feature score based on various MI based scoring 
 * algorithms
 * @author pranab
 *
 */
public class MutualInformation extends Configured implements Tool {
    private static final int CLASS_DIST = 1;
    private static final int FEATURE_DIST = 2;
    private static final int FEATURE_PAIR_DIST = 3;
    private static final int FEATURE_CLASS_DIST = 4;
    private static final int FEATURE_PAIR_CLASS_DIST = 5;
    private static final int FEATURE_CLASS_COND_DIST = 6;
    private static final int FEATURE_PAIR_CLASS_COND_DIST = 7;

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
        job.setCombinerClass(MutualInformation.DistributionCombiner.class);
        
        job.setMapOutputKeyClass(TextTuple.class);
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
	public static class DistributionMapper extends Mapper<LongWritable, Text, TextTuple, Tuple> {
		private String[] items;
		private TextTuple outKey = new TextTuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private FeatureSchema schema;
        private FeatureField classAttrField;
        private List<FeatureField> featureFields;
        private String classAttrVal;
        private String featureAttrVal;
        private String featureAttrBin;
        private String firstFeatureAttrBin;
        private int bin;
        private static final int ONE = 1;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	fieldDelimRegex = context.getConfiguration().get("field.delim.regex", ",");
        	
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "mut.feature.schema.file.path");
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
			context.getCounter("Basic", "Records").increment(1);

            //class
        	outKey.initialize();
        	outVal.initialize();
        	outKey.add(CLASS_DIST,classAttrField.getOrdinal());
			outVal.add(classAttrVal, ONE);
	   		context.write(outKey, outVal);

            //feature and feature class
            for (FeatureField field : featureFields) {
            	outKey.initialize();
            	outVal.initialize();
            	
    			featureAttrVal = items[field.getOrdinal()];
    			setDistrValue(field);

    			//feature
    			outKey.add(FEATURE_DIST, field.getOrdinal());
    			outVal.add(featureAttrBin, ONE);
   	   			context.write(outKey, outVal);
   	   			
   	   			//feature class
            	outKey.initialize();
  	   			outVal.initialize();
  	   		   	outKey.add(FEATURE_CLASS_DIST, field.getOrdinal());
   	   			outVal.add(featureAttrBin, classAttrVal, ONE);
   	   			context.write(outKey, outVal);
   	   			
   	   			//feature class conditional
   	           	outKey.initialize();
 	   			outVal.initialize();
 	   		   	outKey.add(FEATURE_CLASS_COND_DIST, field.getOrdinal(), classAttrVal);
   	   			outVal.add(featureAttrBin,  ONE);
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
         			outVal.add(firstFeatureAttrBin, featureAttrBin, ONE);
       	   			context.write(outKey, outVal);
       	   			
       	   			//feature pair class 
                	outKey.initialize();
                	outVal.initialize();
        			outKey.add(FEATURE_PAIR_CLASS_DIST, featureFields.get(i).getOrdinal(), 
        					featureFields.get(j).getOrdinal());
         			outVal.add(firstFeatureAttrBin, featureAttrBin, classAttrVal, ONE);
       	   			context.write(outKey, outVal);
       	   			
       	   			
       	   			//feature pair class conditional
                	outKey.initialize();
                	outVal.initialize();
       	   			outKey.add(FEATURE_PAIR_CLASS_COND_DIST, featureFields.get(i).getOrdinal(), 
        					featureFields.get(j).getOrdinal(), classAttrVal);
         			outVal.add(firstFeatureAttrBin, featureAttrBin, ONE);
       	   			context.write(outKey, outVal);
            	}
            }

        }     
        
    	/**
    	 * sets feature attribute  bin value
    	 * @param field
    	 */
    	private void setDistrValue(FeatureField field) {
    		if  (field.isCategorical()) {
    			featureAttrBin= featureAttrVal;
    		} else if (field.isInteger()){
    			bin = Integer.parseInt(featureAttrVal) / field.getBucketWidth();
    			featureAttrBin = "" + bin;
    		} else if (field.isDouble()){
    			bin = (int)(Double.parseDouble(featureAttrVal) / field.getBucketWidth());
    			featureAttrBin = "" + bin;
    		} else {
    			throw new IllegalStateException("invalid data type");
    		}
    	}
        
	}

	/**
	 * Combiner
	 * @author pranab
	 *
	 */
	public static class DistributionCombiner extends Reducer<TextTuple, Tuple, TextTuple, Tuple> {
		private Tuple outVal = new Tuple();
		private String attrValue;
		private int attrCount;
		private Integer curCount;
		private Map<String,Integer> distr = new HashMap<String,Integer>();
		private String secondAttrValue;
		private Pair<String, String> attrValuePair;
		private Map<Pair<String, String>, Integer> jointDistr = new HashMap<Pair<String, String>, Integer>();
		private Tuple attrPairClassValue;
		private Map<Tuple, Integer> jointClassdistr = new HashMap<Tuple, Integer>();
		private static final Logger LOG = Logger.getLogger(MutualInformation.DistributionCombiner.class);
				
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
             	LOG.setLevel(Level.DEBUG);
             	System.out.println("in debug mode");
            }
        }		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void reduce(TextTuple key, Iterable<Tuple> values, Context context)
            	throws IOException, InterruptedException {
			key.prepareForRead();
	   		int distrType = key.getInt(0);
   			distr.clear();
   			jointDistr.clear();
   			jointClassdistr.clear();
   			LOG.debug("key:" + key.toString()  + " distrType:" + distrType);
   			
	   		if (distrType == CLASS_DIST) {
	   			//class
	   			populateDistrMap(values);
	   			emitDistrMap(key,  context);
	   		} else if (distrType == FEATURE_DIST) {
	   			//feature
	   			populateDistrMap(values);
	   			emitDistrMap(key,  context);
	   		} else if (distrType == FEATURE_PAIR_DIST) {
	   			//feature pair
	   			populateJointDistrMap(values);
	   			emitJointDistrMap(key,  context);
	   		}  else if (distrType == FEATURE_CLASS_DIST) {
	   			//feature class
	   			populateJointDistrMap(values);
	   			emitJointDistrMap(key,  context);
	   		} else if (distrType == FEATURE_PAIR_CLASS_DIST) {
	   			//feature pair class
	   			populateJointClassDistrMap(values);
	   			emitJointClassDistrMap(key,  context);
	   		} else if (distrType == FEATURE_CLASS_COND_DIST) {
	   			//feature class conditional
	   			populateDistrMap(values);
	   			emitDistrMap(key,  context);
	   		}else if (distrType == FEATURE_PAIR_CLASS_COND_DIST) {
	   			//feature pair class conditional
	   			populateJointDistrMap(values);
	   			emitJointDistrMap(key,  context);
	   		}
		}
		
	   	/**
	   	 * Populates distribution map
	   	 * @param values
	   	 * @param distr
	   	 */
	   	private void populateDistrMap(Iterable<Tuple> values) {
   			for (Tuple value : values) {
   				//LOG.debug(value.toString());
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
	   	 * @param key
	   	 * @param context
	   	 * @throws IOException
	   	 * @throws InterruptedException
	   	 */
	   	private void emitDistrMap(TextTuple key,  Context context) throws IOException, InterruptedException {
   			for (String val : distr.keySet()) {
   				outVal.initialize();
   				outVal.add(val, distr.get(val));
   		   		context.write(key, outVal);
   			}
	   		
	   	}
	   	
	   	/**
	   	 * Populates joint distribution map
	   	 * @param values
	   	 * @param distr
	   	 */
	   	private void populateJointDistrMap(Iterable<Tuple> values) {
   			for (Tuple value : values) {
   				attrValue = value.getString(0);
   				secondAttrValue = value.getString(1);
   				attrCount = value.getInt(2);
   				attrValuePair = new Pair<String, String>(attrValue, secondAttrValue);
  				   				
  				curCount = jointDistr.get(attrValuePair);
   				if (null == curCount) {
   					jointDistr.put(attrValuePair, attrCount);
   				} else {
   					jointDistr.put(attrValuePair, curCount + attrCount);
   				}
   			}	   		
	   	}

	   	/**
	   	 * @param key
	   	 * @param context
	   	 * @throws IOException
	   	 * @throws InterruptedException
	   	 */
	   	private void emitJointDistrMap(TextTuple key,  Context context) throws IOException, InterruptedException {
   			for (Pair<String,String> valPair  : jointDistr.keySet()) {
   				outVal.initialize();
   				outVal.add(valPair.getLeft(), valPair.getRight(), jointDistr.get(valPair));
   		   		context.write(key, outVal);
   			}	   		
	   	}	   	
	   	
   		/**
   		   * @param values
   		   * @param distr
   		   */
   		 private void populateJointClassDistrMap(Iterable<Tuple> values) {
   	   			for (Tuple value : values) {
   	   				attrCount = value.getInt(3);
   	   				attrPairClassValue = value.subTuple(0, 3);
   	  				   				
   	  				curCount = jointClassdistr.get(attrPairClassValue);
   	   				if (null == curCount) {
   	   					jointClassdistr.put(attrPairClassValue, attrCount);
   	   				} else {
   	   					jointClassdistr.put(attrPairClassValue, curCount + attrCount);
   	   				}
   	   			}	   		
   		  }	   	

 	   	/**
 	   	 * @param key
 	   	 * @param context
 	   	 * @throws IOException
 	   	 * @throws InterruptedException
 	   	 */
 	   	private void emitJointClassDistrMap(TextTuple key,  Context context) throws IOException, InterruptedException {
   			for (Tuple valTuple  : jointClassdistr.keySet()) {
   				outVal.initialize();
   				outVal.add(valTuple.getString(0), valTuple.getString(1), valTuple.getString(2), jointClassdistr.get(valTuple));
   		   		context.write(key, outVal);
   			}	   		
	   	}	   	
	   	
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class DistributionReducer extends Reducer<TextTuple, Tuple, NullWritable, Text> {
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
        private List<FeatureField> featureFields;
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
		private String[] mututalInfoScoreAlgList;
		private MutualInformationScore mutualInformationScore = new MutualInformationScore();
		private double redundancyFactor;
		private boolean featureClassCondDstrSepOutput;
 		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
	    	fieldDelim = conf.get("field.delim.out", ",");

	    	InputStream fs = Utility.getFileStream(context.getConfiguration(), "mut.feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            featureFields = schema.getFeatureAttrFields();

            outputMutualInfo = conf.getBoolean("mut.output.mutual.info", true);
            String mututalInfoScoreAlg =  conf.get("mut.mutual.info.score.algorithms", "mutual.info.maximization");
            mututalInfoScoreAlgList = mututalInfoScoreAlg.split(",");
            redundancyFactor = Double.parseDouble(conf.get("mut.mutual.info.redundancy.factor", "1.0"));
            
            featureClassCondDstrSepOutput = conf.getBoolean("mut.feature.class.cond.dstr.sep.output", false);
		}
		
	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	@Override
	   	protected void cleanup(Context context)  throws IOException, InterruptedException {
	   		//emit distributions
	   		outputDistr(context);
	   		
	   		//emit mutual information
	   		outputMutualInfo(context);
	
	   		//output mutual info based scores
	   		outputMutualInfoScore(context);
	   	}
	   	
	   	/**
	   	 * Outputs distr values
	   	 * @param context
	   	 * @throws IOException
	   	 * @throws InterruptedException
	   	 */
	   	private void outputDistr(Context context) throws IOException, InterruptedException {
	   		//class distr
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
	   		
	   		//feature distr
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
	   		
	   		//feature pair distr
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
	   		
	   		//feature class distr
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
	   		
	   		//feature pair class distr
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
	   		
	   		
	   		//feature class conditional distr separate output
	   		if (featureClassCondDstrSepOutput) {
	   			writeSepOutputFeatureClassDstr(context);
	   		} 
	   		
	   		outVal.set("distribution:featureClassConditional");
	   		context.write(NullWritable.get(), outVal);
	   		for (Pair<Integer, String> featureOrdinalClassVal : allFeatureClassCondDistr.keySet()) {
	   			String classVal = featureOrdinalClassVal.getRight();
	   			Map<String, Integer> featureDistr = allFeatureClassCondDistr.get(featureOrdinalClassVal);
		   		for (String featureVal :  featureDistr.keySet()) {
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrdinalClassVal.getLeft()).append(fieldDelim).
		   				append(featureOrdinalClassVal.getRight()).append(fieldDelim).append(featureVal).append(fieldDelim).
		   				append(((double)featureDistr.get(featureVal)) / classDistr.get(classVal));
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
		   		
		   		}
	   		}
	   		//feature pair class conditional distr
	   		outVal.set("distribution:featurePairClassConditional");
	   		context.write(NullWritable.get(), outVal);
	   		for (Tuple featureOrdinalsClassVal : allFeaturePairClassCondDistr.keySet()) {
	   			String classVal = featureOrdinalsClassVal.getString(2);
	   			Map<Pair<String, String>, Integer> featurePairDistr = 
	   					allFeaturePairClassCondDistr.get(featureOrdinalsClassVal);
	   			for (Pair<String, String> values : featurePairDistr.keySet()) {
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrdinalsClassVal.getInt(0)).append(fieldDelim).
		   				append(featureOrdinalsClassVal.getInt(1)).append(fieldDelim).
		   				append(featureOrdinalsClassVal.getString(2)).append(fieldDelim).
		   				append(values.getLeft()).append(fieldDelim).append(values.getRight()).
		   				append(fieldDelim).append(((double)featurePairDistr.get(values)) / classDistr.get(classVal));
			   		outVal.set(stBld.toString());
			   		context.write(NullWritable.get(), outVal);
	   			}
	   		}
	   		
	   	}
	   	
	   	/**
	   	 * @param context
	   	 * @throws IOException
	   	 */
	   	private void writeSepOutputFeatureClassDstr(Context context) throws IOException  {
	   		//get everything with right structure
	   		Map<Integer, List<Map<String, Integer>>> featureClassCondDistr = 
	   				new HashMap<Integer, List<Map<String, Integer>>>();
	   		Map<Integer, List<String>> featureClassValues = new HashMap<Integer, List<String>>();
	   		Map<String, Integer> additionalClassCounts = new HashMap<String, Integer>();
	   		
	   		for (Pair<Integer, String> featureOrdinalClassVal : allFeatureClassCondDistr.keySet()) {
	   			String classVal = featureOrdinalClassVal.getRight();
	   			int featureOrd = featureOrdinalClassVal.getLeft();
	   			Map<String, Integer> featureDistr = allFeatureClassCondDistr.get(featureOrdinalClassVal);
	   			List<Map<String, Integer>> distrList = featureClassCondDistr.get(featureOrd);
	   			List<String> classValues = featureClassValues.get(featureOrd);
	   			if (null == distrList) {
	   				distrList = new ArrayList<Map<String, Integer>>();
	   				featureClassCondDistr.put(featureOrd, distrList);
	   				
	   				classValues = new ArrayList<String>();
	   				featureClassValues.put(featureOrd, classValues);
	   			}
   				distrList.add(featureDistr);
   				classValues.add(classVal);
	   		}
	   		
        	Configuration config = context.getConfiguration();
            OutputStream os = Utility.getCreateFileOutputStream(config, "mut.feature.class.distr.output.file.path");
	   		for (int featureOrd : featureClassCondDistr.keySet()) {
		   		//Laplace correction
	   			additionalClassCounts.clear();
	   			List<Map<String, Integer>> distrList = featureClassCondDistr.get(featureOrd);
	   			List<String> classValues = featureClassValues.get(featureOrd);
	   			if (distrList.size() != 2) {
	   				throw new IllegalStateException("expecting class attribute");
	   			}
	   			
	   			addMissingFeatureValue(distrList, classValues, additionalClassCounts, 0, 1);
	   			addMissingFeatureValue(distrList, classValues, additionalClassCounts, 1, 0);
	   			
		   		//output
	   			int i = 0;
	   			for (Map<String, Integer> featureDistr : distrList) {
	   				String classVal = classValues.get(i++);
	   				for(String featureVal : featureDistr.keySet()) {
	   					stBld.delete(0, stBld.length());
	   					int count = classDistr.get(classVal) + (additionalClassCounts.containsKey(classVal) ? 
	   							additionalClassCounts.get(classVal) : 0);
	   					double distr = ((double)featureDistr.get(featureVal)) / count;
			   			stBld.append(featureOrd).append(fieldDelim).
		   					append(classVal).append(fieldDelim).append(featureVal).append(fieldDelim).
		   					append(distr).append("\n");
			   			byte[] data = stBld.toString().getBytes();
			   			os.write(data);
	   					
	   				}
	   			}
	   			
	   		}
            os.flush();
            os.close();
	   	}
	   	
	   	/**
	   	 * @param distrList
	   	 * @param classValues
	   	 * @param additionalClassCounts
	   	 * @param thisDistr
	   	 * @param thatDistr
	   	 */
	   	private void addMissingFeatureValue(List<Map<String, Integer>> distrList, List<String> classValues,
	   			Map<String, Integer> additionalClassCounts, int thisDistr, int thatDistr) {
			Map<String, Integer> otherDistr = distrList.get(thatDistr);
			String otherClass = classValues.get(thatDistr);
   			for (String featureVal : distrList.get(thisDistr).keySet()) {
   				if (!otherDistr.containsKey(featureVal)) {
   					otherDistr.put(featureVal, 1);
   					Integer classCount = additionalClassCounts.get(otherClass);
   					if (null == classCount) {
   						additionalClassCounts.put(otherClass, 1);
   					} else {
   						additionalClassCounts.put(otherClass, classCount + 1);
   					}
   				}
   			}
	   	}

	   	/**
	   	 * Outputs mutual information value
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
   			double jointFeatureProb;
   			double classProb;
   			Integer count;
   			int numSamples;
   			Tuple featurePairClass = new Tuple();
   			Tuple featureOrdinalsClassCond = new Tuple();

   			//feature class mutual info
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
			   			Pair<String, String> values = new Pair<String, String>(featureVal, classVal);
			   			count = allFeatureClassDistr.get(featureOrd).get(values);
			   			if (null != count) {
			   				jointProb = ((double)count) / totalCount;
			   				sum += jointProb * Math.log(jointProb / (featureProb * classProb));
			   				++numSamples;
			   			}
			   		}
		   		}
		   		double featureClassMuInfo = sum;
		   		if (outputMutualInfo) {
		   			stBld.delete(0, stBld.length());
		   			stBld.append(featureOrd).append(fieldDelim).append(featureClassMuInfo);
		   			outVal.set(stBld.toString());
		   			context.write(NullWritable.get(), outVal);
		   		}
		   		mutualInformationScore.addFeatureClassMutualInfo(featureOrd, featureClassMuInfo);
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
				   			featureProbSecond = ((double)featureDistrSecond.get(featureValSecond)) / totalCount;
				   			Pair<String, String> values = new Pair<String, String>(featureValFirst, featureValSecond);
				   			count = featurePairDistr.get(values);
				   			if (null != count) {
				   				jointProb = ((double)count) / totalCount;
				   				sum += jointProb * Math.log(jointProb / (featureProbFirst * featureProbSecond));
				   				++numSamples;
				   			}
				   		}
			   		}
			   		double featurePairMuInfo = sum;
			   		if (outputMutualInfo) {
			   			stBld.delete(0, stBld.length());
			   			stBld.append(featureOrdinals[i]).append(fieldDelim).append(featureOrdinals[j]).
			   				append(fieldDelim).append(featurePairMuInfo);
				   		outVal.set(stBld.toString());
				   		context.write(NullWritable.get(), outVal);
			   		}
			   		
			   		mutualInformationScore.addFeaturePairMutualInfo(featureOrdinals[i], featureOrdinals[j], featurePairMuInfo);
	   			}
	   		}
	   		
	   		//feature pair class mutual info
	   		outVal.set("mutualInformation:featurePairClass");
	   		context.write(NullWritable.get(), outVal);
	   		for (int i = 0; i < featureOrdinals.length; ++i) {
	   			Map<String, Integer> featureDistrFirst = allFeatureDistr.get(featureOrdinals[i]);
	   			for (int j = i+1; j < featureOrdinals.length; ++j) {
		   			Map<String, Integer> featureDistrSecond = allFeatureDistr.get(featureOrdinals[j]);
	   				Pair<Integer, Integer> featureOrdinalPair = new Pair<Integer, Integer>(featureOrdinals[i], featureOrdinals[j]);
	   				Map<Tuple, Integer> featurePairClassDistr = allFeaturePairClassDistr.get(featureOrdinalPair);
	   				Map<Pair<String, String>, Integer> featurePairDistr = allFeaturePairDistr.get(featureOrdinalPair);
	   				
		   			sum = 0;
		   			numSamples = 0;
		   			double featurePairClassEntropy = 0;
		   			//first feature values
			   		for (String featureValFirst :  featureDistrFirst.keySet()) {
			   			//second feature values
				   		for (String featureValSecond :  featureDistrSecond.keySet()) {
				   			Pair<String, String> values = new Pair<String, String>(featureValFirst, featureValSecond);
				   			
				   			if (null != featurePairDistr.get(values)) {
					   			jointFeatureProb = ((double)featurePairDistr.get(values)) /  totalCount;
					   			//class values
				   				for (String classVal : classDistr.keySet()) {
				   					classProb = ((double)classDistr.get(classVal)) / totalCount;
				   					featurePairClass.initialize();
				   					featurePairClass.add(featureValFirst, featureValSecond, classVal);
				   					count = featurePairClassDistr.get(featurePairClass);
				   					if (null != count) {
						   				jointProb = ((double)count) / totalCount;
						   				sum += jointProb * Math.log(jointProb / (jointFeatureProb * classProb));
						   				
						   				featurePairClassEntropy -=  jointProb * Math.log(jointProb); 
						   				++numSamples;
				   					}
				   				}
				   			}
				   		}
			   		}
			   		double featurePairClassMuInfo = sum;
			   		if (outputMutualInfo) {
			   			stBld.delete(0, stBld.length());
			   			stBld.append(featureOrdinals[i]).append(fieldDelim).append(featureOrdinals[j]).
			   				append(fieldDelim).append(featurePairClassMuInfo);
				   		outVal.set(stBld.toString());
				   		context.write(NullWritable.get(), outVal);
			   		}
			   		mutualInformationScore.addFeaturePairClassMutualInfo(featureOrdinals[i], featureOrdinals[j], featurePairClassMuInfo);
			   		mutualInformationScore.addFeaturePairClassEntropy(featureOrdinals[i], featureOrdinals[j], featurePairClassEntropy);
	   			}
	   		}
	   		
	   		//feature pair class conditional mutual info
	   		outVal.set("mutualInformation:featurePairClassConditional");
	   		context.write(NullWritable.get(), outVal);
	   		for (int i = 0; i < featureOrdinals.length; ++i) {
	   			for (int j = i+1; j < featureOrdinals.length; ++j) {
	   		   		double featurePairClassCondMuInfo = 0;
	   		   		//class values
	   				for (String classVal :  classDistr.keySet()) {
	   					classProb = ((double)classDistr.get(classVal)) / totalCount;
	   					Pair<Integer,String> firstFeatureClassCond = new Pair<Integer,String>(featureOrdinals[i], classVal); 
	   					Map<String, Integer> featureDistrFirst = allFeatureClassCondDistr.get(firstFeatureClassCond);
			   			Pair<Integer,String> secondFeatureClassCond = new Pair<Integer,String>(featureOrdinals[j], classVal); 
			   			Map<String, Integer> featureDistrSecond = allFeatureClassCondDistr.get(secondFeatureClassCond);

			   			if (null == featureDistrFirst || null == secondFeatureClassCond) {
			   				continue;
			   			}
			   			
			   			//joint distr
			   			featureOrdinalsClassCond.initialize();
			   			featureOrdinalsClassCond.add(featureOrdinals[i], featureOrdinals[j], classVal);
			   			Map<Pair<String, String>, Integer> featurePairDistr = 
			   					allFeaturePairClassCondDistr.get(featureOrdinalsClassCond);
			   			sum = 0;
			   			numSamples = 0;
			   			//first feature values
				   		for (String featureValFirst :  featureDistrFirst.keySet()) {
				   			featureProbFirst = ((double)featureDistrFirst.get(featureValFirst)) / totalCount;
				   			//second feature values
					   		for (String featureValSecond :  featureDistrSecond.keySet()) {
					   			featureProbSecond = ((double)featureDistrSecond.get(featureValSecond)) / totalCount;
					   			Pair<String, String> values = new Pair<String, String>(featureValFirst, featureValSecond);
					   			count = featurePairDistr.get(values);
					   			if (null != count) {
					   				jointProb = ((double)count) / totalCount;
					   				sum +=  classProb * (jointProb * Math.log(jointProb / (featureProbFirst * featureProbSecond)));
					   				++numSamples;
					   			}
					   		}			 
				   		}
				   		featurePairClassCondMuInfo += sum;
	   				}
			   		if (outputMutualInfo) {
			   			stBld.delete(0, stBld.length());
			   			stBld.append(featureOrdinals[i]).append(fieldDelim).append(featureOrdinals[j]).
			   				append(fieldDelim).append(featurePairClassCondMuInfo);
				   		outVal.set(stBld.toString());
				   		context.write(NullWritable.get(), outVal);
			   		}
	   			} 
	   		} 
	   		
	   	}
	   	
	   	/**
	   	 * Outputs various mutual information based  scores depending on the configured algorithms
	   	 * @param context
	   	 * @throws IOException
	   	 * @throws InterruptedException
	   	 */
	   	private void outputMutualInfoScore(Context context) throws IOException, InterruptedException {
	   		//all configured algorithms
	   		for (String scoreAlg : mututalInfoScoreAlgList) {
		   		outVal.set("mutualInformationScoreAlgorithm: "+scoreAlg);
		   		context.write(NullWritable.get(), outVal);
		   		if (scoreAlg.equals("mutual.info.maximization")) {
		   			//MIM
		   			List<MutualInformationScore.FeatureMutualInfo>  featureClassMutualInfoList = 
		   					mutualInformationScore.getMutualInfoMaximizerScore();
		   			outputMutualInfoScoreHelper( featureClassMutualInfoList,context);
		   		} else if (scoreAlg.equals("mutual.info.selection")) {
		   			//MIFS
			   		List<MutualInformationScore.FeatureMutualInfo>  featureClassMutualInfoList = 
			   				mutualInformationScore.getMutualInfoFeatureSelectionScore(redundancyFactor);
		   			outputMutualInfoScoreHelper( featureClassMutualInfoList,context);
		   		} else if (scoreAlg.equals("joint.mutual.info")) {
		   			//JMI
			   		List<MutualInformationScore.FeatureMutualInfo>  jointFeatureClassMutualInfoList = 
			   				mutualInformationScore.getJointMutualInfoScore();
		   			outputMutualInfoScoreHelper( jointFeatureClassMutualInfoList,context);
		   		} else if (scoreAlg.equals("double.input.symmetric.relevance")) {
		   			//DISR
			   		List<MutualInformationScore.FeatureMutualInfo>  doubleInputSymmetricalRelevanceList = 
			   				mutualInformationScore.getDoubleInputSymmetricalRelevanceScore();
		   			outputMutualInfoScoreHelper( doubleInputSymmetricalRelevanceList,context);
		   		} else if (scoreAlg.equals("min.redundancy.max.relevance")) {
		   			//MRMR
		   			List<FeatureMutualInfo>  minredMaxRelList = mutualInformationScore.getMinRedundancyMaxrelevanceScore( );
		   			outputMutualInfoScoreHelper( minredMaxRelList,context);
		   		}
	   		}
	   	}

	   	/**
	   	 * @param featureClassMutualInfoList
	   	 * @param context
	   	 * @throws IOException
	   	 * @throws InterruptedException
	   	 */
	   	private void outputMutualInfoScoreHelper(List<MutualInformationScore.FeatureMutualInfo>  featureClassMutualInfoList, 
	   			Context context) throws IOException, InterruptedException {
   			for (MutualInformationScore.FeatureMutualInfo  featureClassMutualInfo :  featureClassMutualInfoList) {
	   			stBld.delete(0, stBld.length());
	   			stBld.append(featureClassMutualInfo.getLeft()).append(fieldDelim).append(featureClassMutualInfo.getRight());
		   		outVal.set(stBld.toString());
		   		context.write(NullWritable.get(), outVal);
   			}
	   		
	   	}
	   	
	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	protected void reduce(TextTuple key, Iterable<Tuple> values, Context context)
            	throws IOException, InterruptedException {
	   		key.prepareForRead();
	   		distrType = key.getInt(0);
	   		if (distrType == CLASS_DIST) {
	   			//class
	   			populateDistrMap(values, classDistr);
	   		} else if (distrType == FEATURE_DIST) {
	   			//feature
	   			int featureOrdinal = key.getInt(1);
	   			Map<String, Integer> featureDistr = allFeatureDistr.get(featureOrdinal);
	   			if (null == featureDistr) {
	   				featureDistr = new HashMap<String, Integer>();
	   				allFeatureDistr.put(featureOrdinal, featureDistr);
	   			}
	   			populateDistrMap(values, featureDistr);
	   		} else if (distrType == FEATURE_PAIR_DIST) {
	   			//feature pair
	   			Pair<Integer, Integer> featureOrdinals = new Pair<Integer, Integer>(key.getInt(1), key.getInt(2));
	   			Map<Pair<String, String>, Integer> featurePairDistr = allFeaturePairDistr.get(featureOrdinals);
	   			if (null == featurePairDistr) {
	   				featurePairDistr = new HashMap<Pair<String, String>, Integer>();
	   				allFeaturePairDistr.put(featureOrdinals, featurePairDistr);
	   			}
	   			populateJointDistrMap(values, featurePairDistr);
	   		} else if (distrType == FEATURE_CLASS_DIST) {
	   			//feature and class
	   			int featureOrdinal = key.getInt(1);
	   			Map<Pair<String, String>, Integer> featureClassDistr = allFeatureClassDistr.get(featureOrdinal);
	   			if (null == featureClassDistr) {
	   				featureClassDistr = new HashMap<Pair<String, String>, Integer>();
	   				allFeatureClassDistr.put(featureOrdinal, featureClassDistr);
	   			}
	   			populateJointDistrMap(values, featureClassDistr);
	   		}else if (distrType == FEATURE_PAIR_CLASS_DIST) {
	   			//feature pair and class
	   			Pair<Integer, Integer> featureOrdinals = new Pair<Integer, Integer>(key.getInt(1), key.getInt(2));
	   			Map<Tuple, Integer> featurePairClassDistr = allFeaturePairClassDistr.get(featureOrdinals);
	   			if (null == featurePairClassDistr) {
	   				featurePairClassDistr = new HashMap<Tuple, Integer>();
	   				allFeaturePairClassDistr.put(featureOrdinals, featurePairClassDistr);
	   			}
	   			populateJointClassDistrMap(values, featurePairClassDistr);
	   		} else if (distrType == FEATURE_CLASS_COND_DIST) {
	   			//feature class conditional
	   			Pair<Integer, String> featureOrdinalClassVal = 
	   					new Pair<Integer, String>(key.getInt(1), key.getString(2));
	   			Map<String, Integer> featureDistr = allFeatureClassCondDistr.get(featureOrdinalClassVal);
	   			if (null == featureDistr) {
	   				featureDistr = new HashMap<String, Integer>();
	   				allFeatureClassCondDistr.put(featureOrdinalClassVal, featureDistr);
	   			}
	   			populateDistrMap(values, featureDistr);
	   		} else if (distrType == FEATURE_PAIR_CLASS_COND_DIST) {
	   			//feature pair class conditional
	   			Tuple featureOrdinalsClassVal = new Tuple();
	   			featureOrdinalsClassVal.add(key.getInt(1), key.getInt(2), key.getString(3));
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
	   	 * Populates distribution map
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
	   	 * Populates joint distribution map
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
