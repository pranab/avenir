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
import java.io.InputStream;

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
import org.avenir.tree.DecisionTreeModel;
import org.chombo.util.BasicUtils;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Utility;

/**
 * Generic classification model predictor MR 
 * @author pranab
 *
 */
public class ModelPredictor extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "model predictor MR";
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
		private PredictiveModel model;
		private EnsemblePredictiveModel ensembleModel;
		private String predClass;
		private String outputMode;
		private int idOrdinal;
		private int classAttrOrdinal;
		private StringBuilder stBld =  new StringBuilder();;
		private static String CLASS_DEC_TREE = "decTreeClassifier";
		private static final String OUTPUT_WITH_RECORD = "withRecord";
		private static final String OUTPUT_WITH_ID = "withKId";
		private static final String OUTPUT_WITH_CLASS_ATTR = "withActualClassAttr";
		

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	fieldDelim = config.get("field.delim.out", ",");

        	//schema
        	FeatureSchema schema = Utility.getFeatureSchema(config, "mop.feature.schema.file.path");
            
            //model files
            String modelDirPath = Utility.assertStringConfigParam(config, "mop.model.dir.path", 
            		"missing model directory path");
            String[] modelFileNames = Utility.assertStringArrayConfigParam(config, "mop.model.file.names", Utility.configDelim,
            		"missing mode file names");
            String classifierType = Utility.assertStringConfigParam(config, "mop.classifier.type", 
            		"missing classifier type");
            
            //error counting
            boolean errorCountingEnabled = config.getBoolean("mop.error.counting.enabled", false);
            int classAttrOrd = -1;
            if (errorCountingEnabled) {
            	classAttrOrd = Utility.assertIntConfigParam(config, "mop.class.attr.ord", "");
            }
 
            //cost based classification
            boolean costBasedPredictionEnabled = config.getBoolean("mop.cost.based.prediction.enabled", false);
            String[] classAttrValues = null;
            double[] misclassCosts = null;
            if (costBasedPredictionEnabled || errorCountingEnabled) {
            	//actual class attribute ordinal
            	classAttrValues = Utility.assertStringArrayConfigParam(config, "mop.class.attr.values", Utility.configDelim,
                		"missing class atrribute values, need for for cost based prediction");
            	if (classAttrValues.length > 2) {
            		throw new IllegalStateException("cost based classification possible only for binary classification");
            	}
            	
            	//make sure about error file path
            	Utility.assertStringConfigParam(config, "map.error.rate.file.path", "missing error file path");
            	
            	//error cost 
            	if (costBasedPredictionEnabled) {
            		misclassCosts = Utility.assertDoubleArrayConfigParam(config, "mop.miss.class.costs", Utility.configDelim, 
            			"missing misclassification costs");
            	}
            }
            
            //build model
            if (modelFileNames.length > 1) {
            	//ensemble
            	double[] memeberWeights = Utility.optionalDoubleArrayConfigParam(config, "mop.ensemble.memeber.weights", 
            			Utility.configDelim);
            	ensembleModel = new EnsemblePredictiveModel();
            	for (int i = 0; i <  modelFileNames.length; ++i) {
            		PredictiveModel memberModel = buildModel(schema, modelDirPath, modelFileNames[i],  classifierType,
                   		 false,  classAttrOrd, classAttrValues, misclassCosts);
            		double weight = null != memeberWeights ? memeberWeights[i] : 1.0;
            		ensembleModel.addModel(memberModel, weight);
            	}
            	
            	//error counting 
                if (errorCountingEnabled) {
                	ensembleModel.enableErrorCounting(classAttrOrd, classAttrValues[0], classAttrValues[1]);
                }
            } else {
            	//single
            	model = buildModel(schema, modelDirPath, modelFileNames[0],  classifierType,
                		 errorCountingEnabled,  classAttrOrd, classAttrValues, misclassCosts);
            }
            
            //output
            outputMode = config.get("mop.output.mode", OUTPUT_WITH_RECORD);
            if (outputMode.equals(OUTPUT_WITH_ID)) {
            	idOrdinal = Utility.assertIntConfigParam(config, "mop.rec.id.ordinal", "missing id ordinal");
            }
            if (outputMode.equals(OUTPUT_WITH_CLASS_ATTR)) {
            	classAttrOrdinal = Utility.assertIntConfigParam(config, "mop.rec.class.attr.ordinal", 
            			"missing class attribute ordinal");
            }
        }   
        
        @Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			super.cleanup(context);
			
			//write error
	        PredictiveModel predModel = null != model ? model : ensembleModel;
			if (predModel.isErrorCountingEnabled()) {
				stBld.delete(0, stBld.length());
				stBld.append("total error: ").append(BasicUtils.formatDouble(predModel.getError())).append("\n").
					append("false positive error: ").append(BasicUtils.formatDouble(predModel.getFalsePosError())).append("\n").
					append("false negative error: ").append(BasicUtils.formatDouble(predModel.getFalseNegError()));
		       	Configuration config = context.getConfiguration();
	            Utility.writeToFile(config, "map.error.rate.file.path", stBld.toString());
			}
        }   
        
        /**
         * @param modelDirPath
         * @param modelFileName
         * @param classifierType
         * @param errorCountingEnabled
         * @param classAttrOrd
         * @param classAttrValues
         * @param misclassCosts
         * @return
         * @throws IOException
         */
        private PredictiveModel buildModel(FeatureSchema schema,String modelDirPath, String modelFileName, String classifierType,
        		boolean errorCountingEnabled, int classAttrOrd, String [] classAttrValues, double[] misclassCosts) throws IOException {
        	PredictiveModel model = null;
        	String modelFilePath = modelDirPath + "/" + modelFileName;
        	InputStream modelStream = Utility.getFileStream(modelFilePath);
        	if (classifierType.equals(CLASS_DEC_TREE)) {
        		model = new DecisionTreeModel(schema, modelStream);
        	} else {
        		throw new IllegalStateException("invalid classifier type");
        	}
    		modelStream.close();
        	
        	//error counting 
            if (errorCountingEnabled) {
            	model.enableErrorCounting(classAttrOrd, classAttrValues[0], classAttrValues[1]);
            }
        	
            //cost based classification
            if (null != misclassCosts) {
            	model.enableCostBasedPrediction(classAttrValues[0], classAttrValues[1], 
            			misclassCosts[0], misclassCosts[1]);
            }
        	return model;
        }
        
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            predClass = null != model ? model.predict(items) : ensembleModel.predict(items);
            
			stBld.delete(0, stBld.length());
            if (outputMode.equals(OUTPUT_WITH_RECORD)) {
            	//full record
            	stBld.append(value.toString()).append(fieldDelim).append(predClass);
            } else  {
            	//partial record
            	if (outputMode.equals(OUTPUT_WITH_ID)) {
            		stBld.append(items[idOrdinal]).append(fieldDelim);
            	} 
            	if (outputMode.equals(OUTPUT_WITH_CLASS_ATTR)) {
            		stBld.append(items[classAttrOrdinal]).append(fieldDelim);
            	} 
             
            	if (stBld.length() == 0) {
            		throw new IllegalStateException("invalid output mode");
            	}
            	stBld.append(predClass);
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
        int exitCode = ToolRunner.run(new ModelPredictor(), args);
        System.exit(exitCode);
    }

}
