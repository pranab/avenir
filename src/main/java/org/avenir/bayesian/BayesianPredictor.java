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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
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
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.util.ConfusionMatrix;
import org.avenir.util.CostBasedArbitrator;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * Predict with naive bayes
 * @author pranab
 *
 */
public class BayesianPredictor extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "bayesian predictor   MR";
        job.setJobName(jobName);
        
        job.setJarByClass(BayesianPredictor.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        
        job.setMapperClass(BayesianPredictor.PredictorMapper.class);

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
        private FeatureSchema schema;
        private List<FeatureField> fields;
        private FeatureField classAttrField;
        private String classAttrVal;
        private String featureAttrVal;
        private Integer featureAttrOrdinal;
        private String featureAttrBin;
        private int featureVal;
        private int bin;
        private BayesianModel model;
		private List<Pair<Integer, Object>> featureValues = new ArrayList<Pair<Integer, Object>>();
		private String[] predictingClasses;
		private String fieldDelim;
		private List<Pair<String, Integer>> classPrediction = new ArrayList<Pair<String,Integer>>();
		private static final int MODEL_DATA_NUM_TOKENS = 4;
		private String predClass;
		private static final String CORRECT = "CORRECT";
		private static final String WRONG = "WRONG";
		private boolean  corrPred;
		private boolean  incorrPred;
		private int predProb;
		private int probThreshHold = 50;
		private  ConfusionMatrix confMatrix;
		private CostBasedArbitrator arbitrator;
		private int classProbDiffThrehold;
		private int classProbDiff;
		private Pair<Integer, Object> feature;
		private boolean outputFeatureProbOnly;
        private double featurePriorProb;
        private Map<String, Double> featurePostProbabilities  = new HashMap<String, Double>();
        private StringBuilder stBld = new StringBuilder();
        private static final Logger LOG = Logger.getLogger(BayesianPredictor.PredictorMapper.class);
       
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
            if (config.getBoolean("debug.on", false)) {
             	LOG.setLevel(Level.DEBUG);
           }
        	
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	fieldDelim = config.get("field.delim.out", ",");

        	//schema
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "bap.feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            fields =  schema.getFeatureAttrFields();
            
            //cost based arbitrator
            if (null != config.get("bap.predict.class.cost")) {
	            String[] costs = config.get("bap.predict.class.cost").split(fieldDelim);
	            arbitrator = new  CostBasedArbitrator(predictingClasses[0], predictingClasses[1], 
	            		Integer.parseInt(costs[0]),  Integer.parseInt(costs[1]));
            }
            
            //class attribute field
        	classAttrField = schema.findClassAttrField();
        	
            //predicting classes and confusion matrix
        	if (null != config.get("bap.predict.class")) {
        		predictingClasses = config.get("bap.predict.class").split(fieldDelim);
        	} else {
            	List<String> cardinality = classAttrField.getCardinality();
            	predictingClasses = new String[2];
        		predictingClasses[0] = cardinality.get(0);
        		predictingClasses[1] = cardinality.get(1);
        	}
    		confMatrix = new ConfusionMatrix(predictingClasses[0], predictingClasses[1] );
    		classProbDiffThrehold = config.getInt("bap.class.prob.diff.threshold", -1);
    		outputFeatureProbOnly  = config.getBoolean("bap.output.feature.prob.only",  false);
        	
        	//bayesian model
        	loadModel(context);
        }
 
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void cleanup(Context context) throws IOException, InterruptedException {
        	if (!outputFeatureProbOnly) {
				context.getCounter("Validation", "TruePositive").increment(confMatrix.getTruePos());
				context.getCounter("Validation", "FalseNegative").increment(confMatrix.getFalseNeg());
				context.getCounter("Validation", "TrueNagative").increment(confMatrix.getTrueNeg());
				context.getCounter("Validation", "FalsePositive").increment(confMatrix.getFalsePos());
				context.getCounter("Validation", "Accuracy").increment(confMatrix.getAccuracy());
				context.getCounter("Validation", "Recall").increment(confMatrix.getRecall());
				context.getCounter("Validation", "Precision").increment(confMatrix.getPrecision());
        	}
        }
         
        /**
         * @param context
         * @throws IOException
         */
        private void loadModel(Context context) throws IOException {
        	model = new BayesianModel();
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "bap.bayesian.model.file.path");
        	BufferedReader reader = new BufferedReader(new InputStreamReader(fs));
        	String line = null; 
        	String[] items = null;
        	
        	while((line = reader.readLine()) != null) {
        		items = line.split(fieldDelimRegex);
    			int featureOrd = !items[1].isEmpty() ? Integer.parseInt(items[1]) : -1;
        		if (items[0].isEmpty()) {
        			//feature prior
        			if (!items[2].isEmpty()) {
        				//binned
        				model.addFeaturePrior(featureOrd, items[2], Integer.parseInt(items[3]));
        			} else {
        				//continuous
        				model.setFeaturePriorParaemeters(featureOrd, Long.parseLong(items[3]), Long.parseLong(items[4]));
        			}
        		} else if (items[1].isEmpty() && items[2].isEmpty()) {
        			//class prior
        			model.addClassPrior(items[0], Integer.parseInt(items[3]));
        		} else {
        			//feature posterior
        			String classVal = items[0];
        			if (!items[2].isEmpty()) {
        				//binned
        				model.addFeaturePosterior(classVal, featureOrd, items[2], Integer.parseInt(items[3]));
        			} else {
        				//continuous
        				model.setFeaturePosteriorParaemeters(classVal, featureOrd, Long.parseLong(items[3]), Long.parseLong(items[4]));
        			}
        		}
        	}
        	
        	//caclulate distributions
        	model.finishUp();
        	
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            classAttrVal = items[classAttrField.getOrdinal()];
            featureValues.clear();
            
            //collect feature attribute and associated bin
        	for (FeatureField field : fields) {
        		if (field.isFeature()) {
        			boolean binned = true;
        			featureAttrOrdinal = field.getOrdinal();
        			featureAttrVal = items[featureAttrOrdinal];
        			if  (field.isCategorical()) {
        				featureAttrBin= featureAttrVal;
        			} else {
        				if (field.isBucketWidthDefined()) {
        					bin = Integer.parseInt(featureAttrVal) / field.getBucketWidth();
        					featureAttrBin = "" + bin;
        				} else {
        					binned = false;
        					featureVal = Integer.parseInt(featureAttrVal);
        				}
        			}
        			if (binned) {
        				feature = new ImmutablePair<Integer, Object>(featureAttrOrdinal, featureAttrBin);
        			} else {
        				feature = new ImmutablePair<Integer, Object>(featureAttrOrdinal, featureVal);
        			}
    				featureValues.add(feature);
        		}
        	}
            
        	//predict probabilty for class values
        	predictClassValue();
        	
        	if (outputFeatureProbOnly) {
        		outputFeatureProb(items[0], context);
        	} else {
        		outputClassPrediction( value,  context);
        	}
        }	

        /**
         * Outputs feature probabilities
         * @param itemID
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void outputFeatureProb(String itemID, Context context)  
            	throws IOException, InterruptedException{
   			stBld.delete(0, stBld.length());
   			stBld.append(itemID).append(fieldDelim).append(featurePriorProb);
            for (String classVal :  predictingClasses) {
            	stBld.append(fieldDelim).append(classVal).append(fieldDelim).append(featurePostProbabilities.get(classVal));
            }   
            stBld.append(fieldDelim).append(classAttrVal);
    		outVal.set(stBld.toString());
			context.write(NullWritable.get(),outVal);
        }
        
        /**
         * @param value
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        private void outputClassPrediction(Text value, Context context)  
        	throws IOException, InterruptedException{
   			stBld.delete(0, stBld.length());
   			if (classPrediction.size() == 1) {
        		//single class
       			predClass = classPrediction.get(0).getLeft();
       			predProb =  classPrediction.get(0).getRight();
       			corrPred = classAttrVal.equals(predClass) && predProb >= probThreshHold;
       			incorrPred = classAttrVal.equals(predClass) && predProb < probThreshHold;
       		    outVal.set(value.toString() + fieldDelim + predClass + fieldDelim + predProb);
        	} else {
        		//all classes
        		if (null != arbitrator){
        			//cost based arbitration
        			 costArbitrate() ;
        		} else {
        			//default arbitration
        			defaultArbitrate();
        		}
       			
       			corrPred = classAttrVal.equals(predClass);
       			incorrPred = !corrPred;
       			confMatrix.report(predClass, classAttrVal);
       			
       			stBld.append(value.toString()).append(fieldDelim).append(predClass).append(fieldDelim).append(predProb);
       			if (classProbDiffThrehold > 0) {
       				stBld.append(fieldDelim);
       				if (classProbDiff > classProbDiffThrehold) {
       					stBld.append("classified");
       				} else {
       					stBld.append("ambiguous");
       				}
       			}
        		outVal.set(stBld.toString());
        	}
        	
        	if (corrPred){
				context.getCounter("Validation", "Correct").increment(1);
        	}
        	if (incorrPred){
				context.getCounter("Validation", "Incorrect").increment(1);
        	}
			context.write(NullWritable.get(),outVal);
        }
        
        /**
         * deafult class artbitrator 
         */
        private void defaultArbitrate() {
    		int prob = 0;
    		String classVal = null;
    		int thisProb;
    		for (Pair<String, Integer> item : classPrediction) {
    			thisProb = item.getRight();
    			if (thisProb > prob) {
    				prob = thisProb;
    				classVal = item.getLeft();
    			}
    		}
    		
    		if (classProbDiffThrehold > 0)  {
    			//difference with next high class probabilty above a threshold
    			classProbDiff = 100;
        		for (Pair<String, Integer> item : classPrediction) {
        			if (!classVal.equals( item.getLeft())) {
        				int diff = prob - item.getRight();
        				if (diff < classProbDiff) {
        					classProbDiff = diff;
        				}
        			}
        		}    			
    		} 
    		
    		//class value with highest probability
    		predClass = classVal;
    		predProb =  prob;
        }
        
        /**
         *  Cost based arbitration
         */
        private void costArbitrate() {
			int posProb = 0;
			int negProb = 0;
			String thisClass;
			int thisProb;
    		for (Pair<String, Integer> item : classPrediction) {
    			thisClass = item.getLeft();
    			thisProb = item.getRight();
    			if (thisClass.equals(predictingClasses[0])) {
    				negProb = thisProb;
    			} else {
    				posProb = thisProb;
    			}
    		}   
    		predClass = arbitrator.arbitrate(posProb, negProb);
    		predProb = 100;
        }
        
        /**
         * feature posterior, class posterior probability
         */
        private void predictClassValue() {
        	double classPriorProb = 0;
        	double featurePostProb = 1.0;
        	int classPostProb = 0;
        	classPrediction.clear();
        	
           	featurePostProbabilities.clear();
			featurePriorProb = model.getFeaturePriorProb(featureValues);
            for (String classVal :  predictingClasses) {
    			classPriorProb = model.getClassPriorProb(classVal);
    			featurePostProb = model.getFeaturePostProb(classVal, featureValues);
    			featurePostProbabilities.put(classVal, featurePostProb);
    			
    			if (classAttrVal.equals(classVal)) {
    				LOG.debug("featurePostProb:" + featurePostProb + " classPriorProb:" + classPriorProb +
    						"featurePriorProb:" + featurePriorProb);
    			}
    			
    			//predict
            	if (!outputFeatureProbOnly) {
            		classPostProb =(int)(((featurePostProb * classPriorProb) / featurePriorProb) * 100);
            		Pair<String, Integer> classProb = new ImmutablePair<String, Integer>(classVal, classPostProb);
            		classPrediction.add(classProb);
            	}
    		}
    	}
    	     
	}
	
	
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new BayesianPredictor(), args);
        System.exit(exitCode);
    }
	
}
