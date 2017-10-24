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

package org.avenir.knn;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.WordUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
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
import org.avenir.knn.Neighborhood.PredictionMode;
import org.avenir.knn.Neighborhood.RegressionMethod;
import org.avenir.util.ConfusionMatrix;
import org.avenir.util.CostBasedArbitrator;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.SecondarySort;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * KNN classifer
 * @author pranab
 *
 */
public class NearestNeighbor extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "K nerest neighbor(KNN)  MR";
        job.setJobName(jobName);
        
        job.setJarByClass(NearestNeighbor.class);
        
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        job.setMapperClass(NearestNeighbor.TopMatchesMapper.class);
        job.setReducerClass(NearestNeighbor.TopMatchesReducer.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        
        job.setGroupingComparatorClass(SecondarySort.TuplePairGroupComprator.class);
        job.setPartitionerClass(SecondarySort.TuplePairPartitioner.class);

        Utility.setConfiguration(job.getConfiguration());

        job.setNumReduceTasks(job.getConfiguration().getInt("num.reducer", 1));
        
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class TopMatchesMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private String trainEntityId;
		private String testEntityId;
		private int rank;
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private String trainClassAttr;
        private String testClassAttr;
        private boolean isValidationMode;
        private String[] items;
        private boolean classCondtionWeighted;
        private double trainingFeaturePostProb;
        private boolean isLinearRegression;
        private String trainRegrNumFld;
        private String testRegrNumFld;
        
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
            fieldDelimRegex = config.get("field.delim.regex", ",");
            isValidationMode = config.getBoolean("nen.validation.mode", true);
            classCondtionWeighted = config.getBoolean("nen.class.condition.weighted", false);
            String predictionMode = config.get("nen.prediction.mode", "classification");
        	String regressionMethod = config.get("nen.regression.method", "average");
        	isLinearRegression = predictionMode.equals("regression") && regressionMethod.equals("linearRegression");
        }    

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            outKey.initialize();
            outVal.initialize();
            if (classCondtionWeighted) {
	            trainEntityId = items[2];
	            testEntityId = items[0];
	            rank = Integer.parseInt(items[3]);
	        	trainClassAttr = items[4];
	        	trainingFeaturePostProb = Double.parseDouble(items[5]); 
	            if (isValidationMode) {
	            	//validation mode
	            	testClassAttr = items[1];
	                outKey.add(testEntityId, testClassAttr, rank);
	            } else {
	            	//prediction mode
	                outKey.add(testEntityId, rank);
	            }
	        	outVal.add(trainEntityId,rank,trainClassAttr,trainingFeaturePostProb);
            } else {
            	int index = 0;
	            trainEntityId = items[index++];
	            testEntityId = items[index++];
	            rank = Integer.parseInt(items[index++]);
	        	trainClassAttr = items[index++];
	            if (isValidationMode) {
	            	//validation mode
	            	testClassAttr = items[index++];
	            } 
	        	outVal.add(trainEntityId,rank,trainClassAttr);
	        	
	        	//for linear regression add numeric input field
	        	if (isLinearRegression) {
	        		trainRegrNumFld = items[index++];
	        		outVal.add(trainRegrNumFld);
	        		
	        		testRegrNumFld = items[index++];
		            if (isValidationMode) {
		                outKey.add(testEntityId, testClassAttr, testRegrNumFld,rank);
		            } else {
		                outKey.add(testEntityId, testRegrNumFld, rank);
		            }
	        		outKey.add(testRegrNumFld);
	        	} else {
		            if (isValidationMode) {
		                outKey.add(testEntityId, testClassAttr, rank);
		            } else {
		                outKey.add(testEntityId, rank);
		            }
	        	}
            }
			context.write(outKey, outVal);
        }
	}
	
    /**
     * @author pranab
     *
     */
    public static class TopMatchesReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
    	private int topMatchCount;
		private String trainEntityId;
		private String testEntityId;
		private int count;
		private int distance;
		private String trainClassValue;
		private Text outVal = new Text();
        private String fieldDelim;
        private boolean isValidationMode;
        private Neighborhood neighborhood;
        private String kernelFunction;
        private int kernelParam;
        private boolean outputClassDistr;
    	private StringBuilder stBld = new StringBuilder();
    	private String testClassValActual;
    	private String testClassValPredicted;
    	private boolean useCostBasedClassifier;
    	private String posClassAttrValue;
    	private String negClassAttrValue;
        private int falsePosCost;
        private int falseNegCost;
        private CostBasedArbitrator costBasedArbitrator;
        private int posClassProbab;
        private boolean classCondtionWeighted;
        private double trainingFeaturePostProb;
        private FeatureSchema schema;
		private  ConfusionMatrix confMatrix;
		private String[] predictingClasses;
	    private FeatureField classAttrField;
		private boolean inverseDistanceWeighted;
		private double decisionThreshold;
        private static final Logger LOG = Logger.getLogger(NearestNeighbor.TopMatchesReducer.class);
       
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
            if (config.getBoolean("debug.on", false)) {
             	LOG.setLevel(Level.DEBUG);
             	System.out.println("in debug mode");
            }
        	
           	fieldDelim = config.get("field.delim", ",");
        	topMatchCount = config.getInt("nen.top.match.count", 10);
            isValidationMode = config.getBoolean("nen.validation.mode", true);
            kernelFunction = config.get("nen.kernel.function", "none");
        	kernelParam = config.getInt("nen.kernel.param", -1);
            classCondtionWeighted = config.getBoolean("nen.class.condtion.weighted", false);
        	neighborhood = new Neighborhood(kernelFunction, kernelParam, classCondtionWeighted);
        	outputClassDistr = config.getBoolean("nen.output.class.distr", false);
        	inverseDistanceWeighted = config.getBoolean("nen.inverse.distance.weighted", false);
        	
        	//regression
        	String predictionMode = config.get("nen.prediction.mode", "classification");
        	if (predictionMode.equals("regression")) {
        		neighborhood.withPredictionMode(PredictionMode.Regression);
            	String regressionMethod = config.get("nen.regression.method", "average");
            	regressionMethod = WordUtils.capitalize(regressionMethod) ;
            	neighborhood.withRegressionMethod(RegressionMethod.valueOf(regressionMethod));
        	}

        	//decision threshold for classification
        	decisionThreshold = Double.parseDouble(config.get("nen.decision.threshold", "-1.0"));
        	if (decisionThreshold > 0 && neighborhood.IsInClassificationMode()) {
            	String[] classAttrValues = config.get("nen.class.attribute.values").split(",");
            	posClassAttrValue = classAttrValues[0];
            	negClassAttrValue = classAttrValues[1];
        		neighborhood.
        			withDecisionThreshold(decisionThreshold).
        			withPositiveClass(posClassAttrValue);
        	}
        	
        	//using cost based arbitrator for classification
        	useCostBasedClassifier = config.getBoolean("nen.use.cost.based.classifier", false);
            if (useCostBasedClassifier && neighborhood.IsInClassificationMode()) {
            	if (null == posClassAttrValue) {
            		String[] classAttrValues = config.get("nen.class.attribute.values").split(",");
            		posClassAttrValue = classAttrValues[0];
            		negClassAttrValue = classAttrValues[1];
            	}
            	
            	int[] missclassificationCost = Utility.intArrayFromString(config.get("nen.misclassification.cost"));
            	falsePosCost = missclassificationCost[0];
            	falseNegCost = missclassificationCost[1];
            	costBasedArbitrator = new CostBasedArbitrator(negClassAttrValue, posClassAttrValue,
            			falseNegCost, falsePosCost);
            }
            
            //confusion matrix for classification validation
       		if (isValidationMode) {
       			if (neighborhood.IsInClassificationMode()) {
	       		    InputStream fs = Utility.getFileStream(context.getConfiguration(), "nen.feature.schema.file.path");
		            ObjectMapper mapper = new ObjectMapper();
		            schema = mapper.readValue(fs, FeatureSchema.class);
		        	classAttrField = schema.findClassAttrField();
		           	List<String> cardinality = classAttrField.getCardinality();
		        	predictingClasses = new String[2];
		    		predictingClasses[0] = cardinality.get(0);
		    		predictingClasses[1] = cardinality.get(1);
		    		confMatrix = new ConfusionMatrix(predictingClasses[0], predictingClasses[1] );
       			}
       		}
            LOG.debug("classCondtionWeighted:" + classCondtionWeighted + "outputClassDistr:" + outputClassDistr);
        }
    	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void cleanup(Context context) throws IOException, InterruptedException {
       		if (isValidationMode) {
       			if (neighborhood.IsInClassificationMode()) {
					context.getCounter("Validation", "TruePositive").increment(confMatrix.getTruePos());
					context.getCounter("Validation", "FalseNegative").increment(confMatrix.getFalseNeg());
					context.getCounter("Validation", "TrueNagative").increment(confMatrix.getTrueNeg());
					context.getCounter("Validation", "FalsePositive").increment(confMatrix.getFalsePos());
					context.getCounter("Validation", "Accuracy").increment(confMatrix.getAccuracy());
					context.getCounter("Validation", "Recall").increment(confMatrix.getRecall());
					context.getCounter("Validation", "Precision").increment(confMatrix.getPrecision());
       			}
        	}
        }
        
    	/* (non-Javadoc)
    	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
    	 */
    	protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
        	if (stBld.length() > 0) {
        		stBld.delete(0,  stBld.length() );
        	}
    		testEntityId  = key.getString(0);
			stBld.append(testEntityId);
			
        	//collect nearest neighbors
    		count = 0;
    		neighborhood.initialize();
        	for (Tuple value : values){
        		int index = 0;
        		trainEntityId = value.getString(index++);
        		distance = value.getInt(index++);
        		trainClassValue = value.getString(index++);
        		if (classCondtionWeighted && neighborhood.IsInClassificationMode()) {
        			trainingFeaturePostProb = value.getDouble(index++);
        			if (inverseDistanceWeighted) {
            			neighborhood.addNeighbor(trainEntityId, distance, trainClassValue,trainingFeaturePostProb, true);
        			} else {
            			neighborhood.addNeighbor(trainEntityId, distance, trainClassValue,trainingFeaturePostProb);
        			}
        		} else {
        			Neighborhood.Neighbor neighbor = neighborhood.addNeighbor(trainEntityId, distance, trainClassValue);
        			if (neighborhood.isInLinearRegressionMode()) {
        				neighbor.setRegrInputVar(Double.parseDouble(value.getString(index++)));
        			}
        		}
        		if (++count == topMatchCount){
        			break;
        		}
			} 
			if (neighborhood.isInLinearRegressionMode()) {
				String testRegrNumFld = isValidationMode? key.getString(2) : key.getString(1);
				neighborhood.withRegrInputVar(Double.parseDouble(testRegrNumFld));
			}
			
			//class distribution
        	neighborhood.processClassDitribution();
			if (outputClassDistr && neighborhood.IsInClassificationMode()) {
	    		if (classCondtionWeighted) {
					 Map<String, Double>  classDistr = neighborhood.getWeightedClassDitribution();
					 double thisScore;
					 for (String classVal : classDistr.keySet()) {
							thisScore = classDistr.get(classVal);
			    			//LOG.debug("classVal:" + classVal + " thisScore:" + thisScore);
				    		stBld.append(fieldDelim).append(classVal).append(fieldDelim).append(thisScore);
					 }
	    		} else {
					Map<String, Integer>  classDistr = neighborhood.getClassDitribution();
				 	int thisScore;
					for (String classVal : classDistr.keySet()) {
						thisScore = classDistr.get(classVal);
		    			 stBld.append(classVal).append(fieldDelim).append(thisScore);
					}
	    		}
			}
			 
    		if (isValidationMode) {
    			//actual class attr value
    	    	testClassValActual  = key.getString(1);
    			stBld.append(fieldDelim).append(testClassValActual);
        	}
    		
    		//predicted class value
    		if (useCostBasedClassifier) {
    			//use cost based arbitrator
    			if (neighborhood.IsInClassificationMode()) {
    				posClassProbab = neighborhood.getClassProb(posClassAttrValue);
    				testClassValPredicted = costBasedArbitrator.classify(posClassProbab);
    			}
    		} else {
    			//get directly
    			if (neighborhood.IsInClassificationMode()) {
    				testClassValPredicted = neighborhood.classify();
    			} else {
    				testClassValPredicted = "" + neighborhood.getPredictedValue();
    			}
    		}
			stBld.append(fieldDelim).append(testClassValPredicted);
			
    		if (isValidationMode) {
    			if (neighborhood.IsInClassificationMode()) {
    				confMatrix.report(testClassValPredicted, testClassValActual);
    			}
    		}    		
 			outVal.set(stBld.toString());
			context.write(NullWritable.get(), outVal);
    	}
    }
 
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new NearestNeighbor(), args);
        System.exit(exitCode);
	}
}
