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

package org.avenir.regress;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.UUID;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
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
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.explore.CramerCorrelation;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;
import org.codehaus.jackson.map.ObjectMapper;

public class LogisticRegressionJob  extends Configured implements Tool {
	private static final String ITER_LIMIT = "iterLimit";
	private static final String ALL_BELOW_THRESHOLD = "allBelowThreshold";
	private static final String AVERAGE_BELOW_THRESHOLD = "averageBelowThreshold";
	private static final int CONVERGED = 100;
	private static final int NOT_CONVERGED = 101;
	
	
	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Logistic regression";
        job.setJobName(jobName);
        job.setJarByClass(LogisticRegressionJob.class);
        Utility.setConfiguration(job.getConfiguration(), "avenir");

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        job.setMapperClass(LogisticRegressionJob.RegressionMapper.class);
        job.setReducerClass(LogisticRegressionJob.RegressionReducer.class);
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setNumReduceTasks(job.getConfiguration().getInt("num.reducer", 1));
        int status =  job.waitForCompletion(true) ? 0 : 1;
        
        Configuration conf = job.getConfiguration();
        if (status == 0) {
        	status = checkConvergence(conf);
        }
        
        return status;
	}
	
	/**
	 * @param conf
	 * @return
	 * @throws IOException
	 */
	private int checkConvergence(Configuration conf) throws IOException {
		int status = 0;
        List<String>   lines = Utility.getFileLines(conf, "coeff.file.path");
		
		String convCriteria =   conf.get("convergence.criteria",  ITER_LIMIT);
		if (convCriteria.equals(ITER_LIMIT)) {
			int iterLimit = conf.getInt("iteration.limit",  10);
	         status = lines.size() < iterLimit  ? NOT_CONVERGED : CONVERGED;
		} else  {
			double[] prevCoeff = Utility.doubleArrayFromString(lines.get(lines.size()-2));
			double[] curCoeff = Utility.doubleArrayFromString(lines.get(lines.size()-1));
			LogisticRegressor regressor = new LogisticRegressor(prevCoeff);
			regressor.setAggregates(curCoeff);
			regressor.setConvergeThreshold((double)conf.getFloat("convergence.threshold", (float)5.0));
			if (convCriteria.equals(ALL_BELOW_THRESHOLD)) {
				status = regressor.isAllConverged() ? CONVERGED : NOT_CONVERGED;
			} else if (convCriteria.equals(AVERAGE_BELOW_THRESHOLD)) {
				status = regressor.isAverageConverged() ? CONVERGED : NOT_CONVERGED;
			} else {
				throw new IllegalArgumentException("Invalid convergence criteria:" + convCriteria);
			}
		} 		
		
		return status;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class RegressionMapper extends Mapper<LongWritable, Text, Text, Tuple> {
		private String fieldDelimRegex;
		private String[] items;
		private Text outKey  = new Text();
		private Tuple outVal  = new Tuple();
        private FeatureSchema schema;
        private int[] featureValues;
        private int[] featureOrdinals;
        private int classOrdinal;
        private String classValue;
        private int  iterCount;
        private double[] coefficients;
        private LogisticRegressor regressor;
        private static final Logger LOG = Logger.getLogger(RegressionMapper.class);
       
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	InputStream fs = Utility.getFileStream(conf, "feature.schema.file.path");
            ObjectMapper mapper = new ObjectMapper();
            schema = mapper.readValue(fs, FeatureSchema.class);
            
            //regression coefficients
            List<String[]>   lines = Utility.parseFileLines(conf, "coeff.file.path", fieldDelimRegex);
            iterCount = lines.size();
            String[] items   = lines.get(lines.size() - 1);
            coefficients = new double[items.length];
            for (int i = 0; i < items.length; ++i) {
            	coefficients[i] = Double.parseDouble(items[i]);
            }
            
            String posClassVal = conf.get("positive.class.value");
            regressor = new  LogisticRegressor(coefficients,  posClassVal);
        }       
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void cleanup(Context context)  throws IOException, InterruptedException {
        	double[] aggregate = regressor.getAggregates();
        	for (int i = 0; i < aggregate.length; ++i) {
        		outVal.append(aggregate[i]);
        	}
        	outKey.set(UUID.randomUUID().toString());
        	context.write(outKey, outVal);
        }
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            if (null == featureValues) {
            	featureOrdinals = schema.getFeatureFieldOrdinals();
            	featureValues = new int[featureOrdinals.length + 1];
            	featureValues[0] = 1;
            	classOrdinal = schema.findClassAttrField().getOrdinal();
            }
            
            for (int i=0;  i <  featureOrdinals.length; ++i) {
            	featureValues[i+1] = Integer.parseInt(items[featureOrdinals[i]]);
            }
            classValue = items[classOrdinal];
            regressor.aggregate(featureValues, classValue);
            
        }       
	}	
	
	/**
	 * @author pranab
	 *
	 */
	public static class RegressionReducer extends Reducer<Text, Tuple, NullWritable, Text> {
        private FeatureSchema schema;
        private LogisticRegressor regressor; 
        private double[] aggregate;
        private String fieldDelimOut;
        private Text outVal = new Text();
        
	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	fieldDelimOut = conf.get("field.delim.out", ",");
	   	}
	   	
	    /* (non-Javadoc)
	     * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
	     */
	    protected void cleanup(Context context)  throws IOException, InterruptedException {
	    	  double[] aggregates = regressor.getAggregates();
	    	  StringBuilder stBld = new StringBuilder();
	    	  stBld.append(aggregates[0]);
	    	  for (int i = 1; i < aggregates.length; ++i ) {
	    		  stBld.append(fieldDelimOut).append(aggregates[i]);
	    	  }
	    	  outVal.set(stBld.toString());
	    	  
	         Configuration conf = context.getConfiguration();
	         saveCoefficients( conf,  stBld.toString());
	     }
	      
	    /**
	     * @param conf
	     * @param newCoefficients
	     * @throws IOException
	     */
	    private void saveCoefficients(Configuration conf, String newCoefficients) throws IOException {
	         List<String>   lines = Utility.getFileLines(conf, "coeff.file.path");
	         lines.add(newCoefficients);
	         
	         //delete file
	         FileSystem fs = FileSystem.get(conf); 
	         Path filenamePath = new Path(conf.get("coeff.file.path"));     	
	         fs.delete(filenamePath, true);
	         
	         //recreate with new data
	         OutputStream os = fs.create( filenamePath);
	         BufferedWriter br = new BufferedWriter( new OutputStreamWriter( os, "UTF-8" ) );
	         for (String line : lines) {
	        	 br.write(line + "\n");
	         }
	         br.close();
	         fs.close();
	    }
	    
	    
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Text  key, Iterable<Tuple> values, Context context)
        throws IOException, InterruptedException {
    		for (Tuple value : values) {
    			if (null == regressor) {
    				regressor = new LogisticRegressor();
    				aggregate = new double[value.getSize()];
    			}
    			for (int i = 0; i < value.getSize(); ++i) {
    				aggregate[i] = value.getDouble(i);
    			}
    			regressor.addAggregates(aggregate);
    		}
        }	   	
	}	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = NOT_CONVERGED;
        int iterCount = 1;
        do {
        	System.out.println("job iteration count:" + iterCount);
        	exitCode = ToolRunner.run(new LogisticRegressionJob(), args);
        	++iterCount;
        } while (exitCode == NOT_CONVERGED);
        
        System.exit(exitCode);
	}
	
}
