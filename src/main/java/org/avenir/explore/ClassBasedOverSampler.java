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
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.stats.ExponentialDistrRejectionSampler;
import org.chombo.util.BasicUtils;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Utility;

/**
 * Class based over sampling bt SMOTE algorithms
 * @author pranab
 *
 */
public class ClassBasedOverSampler extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Over sampling class balancer ";
        job.setJobName(jobName);
        
        job.setJarByClass(ClassBasedOverSampler.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(ClassBasedOverSampler.SamplingMapper.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(0);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class SamplingMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
		private Text outVal = new Text();
        private String fieldDelimRegex;
        private String fieldDelim;
		private String[] items;
		private String[] srcRec;
		private int recLen;
		private List<String[]> neighborRecs = new ArrayList<String[]>();
		private int beg;
		private int end;
		private int overSamplingMultiplier;
		private FeatureSchema schema;
		private int outputPrecision;
		private String neighborSamplingDistr;
		private ExponentialDistrRejectionSampler expSampler;
		private final String UNIFORM_DISTR = "uniform";
		private final String EXP_DISTR = "exponential";
		
		
	    /* (non-Javadoc)
	     * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
	     */
	    protected void setup(Context context) throws IOException, InterruptedException {
	    	Configuration config = context.getConfiguration();
           	fieldDelim = config.get("field.delim", ",");
            fieldDelimRegex = config.get("field.delim.regex", ",");
            
            //record length
	    	recLen = Utility.assertIntConfigParam(config, "cbos.rec.len", "missing record length");
	    	srcRec = new String[recLen];
	    	
	    	//over sampling multiplier
	    	overSamplingMultiplier = Utility.assertIntConfigParam(config, "cbos.over.sampling.multiplier", 
	    			"missing over sampling multiplier");
	    	
	    	//neighbor sampling distribution
	    	neighborSamplingDistr = config.get("cbos.neighbor.sampling.distr", UNIFORM_DISTR);
	    	if (neighborSamplingDistr.equals(EXP_DISTR)) {
	    		double mean = Utility.assertDoubleConfigParam(config, "cbos.exp.distr.mean", "missing exponential distribution mean");
	    		expSampler = new  ExponentialDistrRejectionSampler(mean);
	    	}
	    	
        	//schema
            schema = Utility.getFeatureSchema(config, "cbos.feature.schema.file.path");
            
            //output precision
            outputPrecision = config.getInt("cbos.output.precision", 3);
	    }
	    
	    @Override
	    protected void map(LongWritable key, Text value, Context context)
	    	throws IOException, InterruptedException {
	    	
	    	items  =  value.toString().split(fieldDelimRegex, -1);
	    	neighborRecs.clear();
	    	
	    	//source rec
	    	beg = 0;
	    	end = beg + recLen;
	    	BasicUtils.arrayCopy(items, beg, end, srcRec);
	    	
	    	//neighbors
	    	for (beg += recLen, end += recLen ; beg < items.length; beg += recLen, end += recLen) {
	    		String[] neighborRec = new String[recLen];
	    		BasicUtils.arrayCopy(items, beg, end, neighborRec);
	    		neighborRecs.add(neighborRec);
	    	}
	    	
	    	//over sample
    		String[] neighborRec = null;
	    	for (int i = 0 ; i < overSamplingMultiplier; ++i) {
	    		if (neighborSamplingDistr.equals(UNIFORM_DISTR)) {
	    			neighborRec = BasicUtils.selectRandom(neighborRecs);
	    		} else  if (neighborSamplingDistr.equals(EXP_DISTR)) {
	    			int neighborSize =  neighborRecs.size() ;
	    			expSampler.setRange(1, neighborSize);
	    			int index = (int)(Math.round(expSampler.sample())) - 1 ;
	    			index = BasicUtils.between(index, 0, neighborSize -1); 
	    			neighborRec = neighborRecs.get(index);
	    		} else {
	    			throw new IllegalStateException("invalid neighbor sampling distribution");
	    		}
	    		String[] newRec = generateSample(srcRec, neighborRec);
	    		outVal.set(BasicUtils.join(newRec, fieldDelim));
				context.write(NullWritable.get(), outVal);
	    	}
	    }
	    
	    /**
	     * @param srcRec
	     * @param neighborRec
	     * @return
	     */
	    private String[] generateSample(String[] srcRec, String[] neighborRec) {
	    	String[] newRec = new String[recLen];
	    	for (int i = 0; i < recLen; ++i) {
	    		FeatureField field = schema.findFieldByOrdinal(i);
	    		if (field.isId()) {
	    			//generate Id
	    			String scrambled = BasicUtils.scramble(srcRec[i] + neighborRec[i] , 6);
	    			newRec[i] = scrambled.substring(0, srcRec[i].length());
	    		} else if (field.isFeature()) {
	    			//feature value
	    			if (field.isInteger()) {
	    				//interpolate
	    				int srVal = Integer.parseInt(srcRec[i]);
	    				int neVal = Integer.parseInt(neighborRec[i]);
	    				int newVal = srVal + (int)((neVal - srVal) * Math.random());
	    				newRec[i] = "" + newVal;
	    			} else if (field.isDouble()) {
	    				//interpolate
	    				double srVal = Double.parseDouble(srcRec[i]);
	    				double neVal = Double.parseDouble(neighborRec[i]);
	    				double newVal = srVal + (neVal - srVal) * Math.random();
	    				newRec[i] = BasicUtils.formatDouble(newVal, outputPrecision);
	    			} else if (field.isCategorical()) {
	    				//choose randomly
	    				newRec[i] = Math.random() < 0.5 ? srcRec[i] : neighborRec[i];
	    			}
	    		} else {
	    			//class value
	    			newRec[i] = srcRec[i];
	    		}
	    	}
	    	return newRec;
	    }
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new ClassBasedOverSampler(), args);
        System.exit(exitCode);
	}

}
