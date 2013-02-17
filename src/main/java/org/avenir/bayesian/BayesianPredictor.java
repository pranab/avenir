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
import java.util.List;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
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
import org.apache.hadoop.util.ToolRunner;
import org.avenir.util.ConfusionMatrix;
import org.avenir.util.CostBasedArbitrator;
import org.chombo.mr.FeatureField;
import org.chombo.mr.FeatureSchema;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

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
        private int bin;
        private BayesianModel model;
		private List<Pair<Integer, String>> featureValues = new ArrayList<Pair<Integer, String>>();
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
		
        
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("bs.field.delim.regex", ",");
        	fieldDelim = config.get("field.delim.out", ",");

        	//schema
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            
            
            //cost based arbitrator
            if (null != config.get("bp.predict.class.cost")) {
	            String[] costs = config.get("bp.predict.class.cost").split(fieldDelim);
	            arbitrator = new  CostBasedArbitrator(predictingClasses[0], predictingClasses[1], 
	            		Integer.parseInt(costs[0]),  Integer.parseInt(costs[1]));
            }
            
            //class attribute field
        	fields = schema.getFields();
        	for (FeatureField field : fields) {
        		if (!field.isFeature()  &&  !field.isId()) {
        			classAttrField = field;
        			break;
        		}
        	}
        	
            //predicting classes and confusion matrix
        	if (null != config.get("bp.predict.class")) {
        		predictingClasses = config.get("bp.predict.class").split(fieldDelim);
        	} else {
            	List<String> cardinality = classAttrField.getCardinality();
            	predictingClasses = new String[2];
        		predictingClasses[0] = cardinality.get(0);
        		predictingClasses[1] = cardinality.get(1);
        	}
    		confMatrix = new ConfusionMatrix(predictingClasses[0], predictingClasses[1] );
        	
        	//bayesian model
        	loadModel(context);
        }
 
        protected void cleanup(Context context) throws IOException, InterruptedException {
			context.getCounter("Validation", "TruePositive").increment(confMatrix.getTruePos());
			context.getCounter("Validation", "FalseNegative").increment(confMatrix.getFalseNeg());
			context.getCounter("Validation", "TrueNagative").increment(confMatrix.getTrueNeg());
			context.getCounter("Validation", "FalsePositive").increment(confMatrix.getFalsePos());
			context.getCounter("Validation", "Recall").increment(confMatrix.getRecall());
			context.getCounter("Validation", "Precision").increment(confMatrix.getPrecision());
        }
         
        private void loadModel(Context context) throws IOException {
        	model = new BayesianModel();
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "bayesian.model.file.path");
        	BufferedReader reader = new BufferedReader(new InputStreamReader(fs));
        	String line = null; 
        	String[] items = null;
        	
        	while((line = reader.readLine()) != null) {
        		items = line.split(fieldDelimRegex);
        		if(items.length != MODEL_DATA_NUM_TOKENS) {
        			throw new IOException("invalid model data");
        		}
        		
        		if (items[0].isEmpty()) {
        			//feature prior
        			model.addFeaturePrior(Integer.parseInt(items[1]), items[2], Integer.parseInt(items[3]));
        		} else if (items[1].isEmpty() && items[2].isEmpty()) {
        			//class prior
        			model.addClassPrior(items[0], Integer.parseInt(items[3]));
        		} else {
        			//feature posterior
        			model.addFeaturePosterior(items[0], Integer.parseInt(items[1]), items[2], Integer.parseInt(items[3]));
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
        			featureAttrOrdinal = field.getOrdinal();
        			featureAttrVal = items[featureAttrOrdinal];
        			if  (field.isCategorical()) {
        				featureAttrBin= featureAttrVal;
        			} else {
        				bin = Integer.parseInt(featureAttrVal) / field.getBucketWidth();
        				featureAttrBin = "" + bin;
        			}
        			Pair<Integer, String> feature = new ImmutablePair<Integer, String>(featureAttrOrdinal, featureAttrBin);
        			featureValues.add(feature);
        		}
        	}
            
        	//predict probabilty for class values
        	predictClassValue();
        	
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
        		outVal.set(value.toString() + fieldDelim + predClass + fieldDelim + predProb);
        	}
        	
        	if (corrPred){
				context.getCounter("Validation", "Correct").increment(1);
        	}
        	if (incorrPred){
				context.getCounter("Validation", "Incorrect").increment(1);
        	}
			context.write(NullWritable.get(),outVal);
        	
        }	
        
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
    		predClass = classVal;
   			predProb =  prob;
        }
        
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
        
        private void predictClassValue() {
        	double classPriorProb = 0;
        	double featurePriorProb = 1.0;
        	double featurePostProb = 1.0;
        	int classPostProb = 0;
        	classPrediction.clear();
        	
    		for (String classVal :  predictingClasses) {
    			classPriorProb = model.getClassPriorProb(classVal);
    			featurePriorProb = model.getFeaturePriorProb(featureValues);
    			featurePostProb = model.getFeaturePostProb(classVal, featureValues);
    			
    			if (classAttrVal.equals(classVal)) {
    				System.out.println("featurePostProb:" + featurePostProb + " classPriorProb:" + classPriorProb +
    						"featurePriorProb:" + featurePriorProb);
    			}
    			
    			//predict
    			classPostProb =(int)(((featurePostProb * classPriorProb) / featurePriorProb) * 100);
    			Pair<String, Integer> classProb = new ImmutablePair<String, Integer>(classVal, classPostProb);
    			classPrediction.add(classProb);
    		}
    	}
    	     
	}
	
	
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new BayesianPredictor(), args);
        System.exit(exitCode);
    }
	
}
