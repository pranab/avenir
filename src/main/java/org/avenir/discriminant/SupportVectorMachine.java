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


package org.avenir.discriminant;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
import org.avenir.tree.DecisionTreeBuilder;
import org.chombo.util.BasicUtils;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * @author pranab
 *
 */
public class SupportVectorMachine extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Support vector machine";
        job.setJobName(jobName);
        job.setJarByClass(SupportVectorMachine.class);
        Utility.setConfiguration(job.getConfiguration(), "avenir");

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
       
        job.setMapperClass(SupportVectorMachine.VectorMapper.class);
        job.setReducerClass(SupportVectorMachine.VectorReducer.class);
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("svm.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class VectorMapper extends Mapper<LongWritable, Text, Text, Tuple> {
		private String fieldDelimRegex;
		private String[] items;
        private Tuple outVal = new Tuple();
		private Text outKey  = new Text();
        private FeatureSchema schema;
        private int[] featureFieldordinals;
        private List<double[]> data = new ArrayList<double[]>();
        private int vectorSize;
        private int classAttrOrdinal;
        private SequentialMinimalOptimization optimizer;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	
        	//schema
            schema = Utility.getFeatureSchema(config, "svm.feature.schema.file.path");
            
            //sorted feature field ordinals
            featureFieldordinals = schema.getFeatureFieldOrdinals();
            vectorSize = featureFieldordinals.length + 2;
            classAttrOrdinal = schema.findClassAttrField().getOrdinal();
            
            double penaltyFactor = (double)config.getFloat("svm.pnalty.factor", (float)0.05);
            double tolerance = (double)config.getFloat("svm.tolerance", (float)0.001);
            double eps = (double)config.getFloat("svm.eps", (float)0.001);
            String kernelType = config.get("svm.kernel.type", SequentialMinimalOptimization.KERNEL_LINER);
            optimizer = new SequentialMinimalOptimization(penaltyFactor, classAttrOrdinal, tolerance, 
        			eps, kernelType, featureFieldordinals.length);
            
            outKey.set(BasicUtils.generateId());
        } 
        
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
		 */
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			optimizer.process(data);
			List<Integer> supVecs = optimizer.getSupVecIndexes();
			for (int index : supVecs) {
				outVal.initialize();
				double[] vec = data.get(index);
				outVal.fromArray(vec);
				context.write(outKey, outVal);
			}
			super.cleanup(context);
		}
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex, -1);
            double[] vector = new double[vectorSize];
            
            //features followed by class attr and lagrangian
            int k = 0;
            for (int i : featureFieldordinals) {
            	vector[k++] = Double.parseDouble(items[i]);
            }
            vector[k++] = Double.parseDouble(items[classAttrOrdinal]);
            vector[k++] = 0;
            data.add(vector);
        }
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class VectorReducer extends Reducer<Text, Tuple, NullWritable, Text> {
		private String fieldDelim;
		private Text outVal  = new Text();
        private List<double[]> data = new ArrayList<double[]>();
        private FeatureSchema schema;
        private SequentialMinimalOptimization optimizer;

	   	@Override
	   	protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelim = config.get("field.delim.out", ",");
        	
        	//schema
            schema = Utility.getFeatureSchema(config, "svm.feature.schema.file.path");
            int[] featureFieldordinals = schema.getFeatureFieldOrdinals();
        	int  classAttrOrdinal = schema.findClassAttrField().getOrdinal();

        	//optimizer
            double penaltyFactor = (double)config.getFloat("svm.pnalty.factor", (float)0.05);
            double tolerance = (double)config.getFloat("svm.tolerance", (float)0.001);
            double eps = (double)config.getFloat("svm.eps", (float)0.001);
            String kernelType = config.get("svm.kernel.type", SequentialMinimalOptimization.KERNEL_LINER);
            optimizer = new SequentialMinimalOptimization(penaltyFactor, classAttrOrdinal, tolerance, 
        			eps, kernelType, featureFieldordinals.length);
	   	}	
	   	
	   	@Override
	   	protected void cleanup(Context context)  throws IOException, InterruptedException {
			optimizer.process(data);
			List<Integer> supVecs = optimizer.getSupVecIndexes();
			for (int index : supVecs) {
				double[] vec = data.get(index);
				outVal.set(BasicUtils.join(vec, fieldDelim));
				context.write(NullWritable.get(), outVal);
			}
			super.cleanup(context);
	   		
	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Text  key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
    		for (Tuple val : values) {
    			double[] vec = val.toDoubleArray();
    			vec[vec.length -1] = 0;
    			data.add(vec);
    		}
        }	   	
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new SupportVectorMachine(), args);
        System.exit(exitCode);
	}
	
}
