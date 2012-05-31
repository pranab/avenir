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

	public static class PredictorMapper extends Mapper<LongWritable, Text, Tuple, IntWritable> {
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
        private BayesianModel model;
		private List<Pair<Integer, String>> featureValues = new ArrayList<Pair<Integer, String>>();
		private String[] predictingClasses;
		private String fieldDelim;
		private List<Pair<String, Integer>> classPrediction = new ArrayList<Pair<String,Integer>>();
        
        protected void setup(Context context) throws IOException, InterruptedException {
        	fieldDelimRegex = context.getConfiguration().get("bs.field.delim.regex", ",");
        	fieldDelim = context.getConfiguration().get("field.delim.out", ",");

        	//schema
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            
            //predicting classes
            predictingClasses = context.getConfiguration().get("bp.predict.class").split(fieldDelim);
            
            //class attribute field
        	fields = schema.getFields();
        	for (FeatureField field : fields) {
        		if (!field.isFeature()) {
        			classAttrField = field;
        			break;
        		}
        	}
        	
        	//bayesian model
        	loadModel(context);
        }
 
        private void loadModel(Context context) throws IOException {
        	model = new BayesianModel();
        	InputStream fs = Utility.getFileStream(context.getConfiguration(), "bayesian.model.file.path");
        	BufferedReader reader = new BufferedReader(new InputStreamReader(fs));
        	String line = null; 
        	String[] items = null;
        	
        	while((line = reader.readLine()) != null) {
        		items = line.split(fieldDelimRegex);
        		if (items[0].isEmpty()) {
        			//feature prior
        			model.addFeaturePrior(Integer.parseInt(items[1]), items[2], Integer.parseInt(items[3]));
        		} else if (items[1].isEmpty()) {
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
        			Pair<Integer, String> feature = new ImmutablePair<Integer, String>(featureAttrOrdinal, featureAttrBin);
        			featureValues.add(feature);
        		}
        	}
            
        	//predict
        	predict();
        	
        }	
        
        private void predict() {
    		
    	}
    	     
	}
	
	
    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new BayesianPredictor(), args);
        System.exit(exitCode);
    }
	
}
