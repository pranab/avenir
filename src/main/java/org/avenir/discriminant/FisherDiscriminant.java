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
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.mr.NumericalAttrStats;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Fisher univariate discriminant. Feature is numeric. Classification is binary
 * @author pranab
 *
 */
public class FisherDiscriminant  extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Univariate Fisher linear discriminant";
        job.setJobName(jobName);
        
        job.setJarByClass(FisherDiscriminant.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(NumericalAttrStats.StatsMapper.class);
        job.setReducerClass(FisherDiscriminant.FisherReducer.class);
        job.setCombinerClass(NumericalAttrStats.StatsCombiner.class);
        
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
	public static class FisherReducer  extends NumericalAttrStats.StatsReducer  {
		private static final String uncondAttrVal = "0";
		private Map<Integer, ConditionedFeatureStat[]> attrCondStats  = new HashMap<Integer, ConditionedFeatureStat[]>();

	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	protected void cleanup(Context context)  throws IOException, InterruptedException {
	   		//emit class boundary
	   		for (int attr : attrCondStats.keySet()) {
	   			ConditionedFeatureStat[] condStats = attrCondStats.get(attr);
	   			double pooledVariance = (condStats[0].getVariance() * condStats[0].getCount() + 
	   				condStats[1].getVariance() * condStats[1].getCount()) / (condStats[0].getCount() + condStats[1].getCount());
	   			double logOddsPrior = Math.log((double)condStats[0].getCount() / condStats[1].getCount());
	   			double meanDiff = condStats[0].getMean() - condStats[1].getMean();
	   			double discrimValue = (condStats[0].getMean() + condStats[1].getMean()) / 2;
	   			discrimValue -= logOddsPrior * pooledVariance / meanDiff;
	   			outVal.set("" + attr +  fieldDelim + logOddsPrior + fieldDelim + pooledVariance + fieldDelim + discrimValue);
	   			context.write(NullWritable.get(), outVal);
	   		}
	   	}	
	   	
    	protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
            	throws IOException, InterruptedException {
    		processReduce(values);

    		//process conditional stats
    		String condAttrVal = key.getString(1);
    		if (!uncondAttrVal.equals(condAttrVal)) {
    			Integer attr = key.getInt(0);
    			ConditionedFeatureStat[] condStats = attrCondStats.get(attr);
    			if (null == condStats) {
    				condStats = new ConditionedFeatureStat[2];
    				condStats[0] = condStats[1] = null;
    				attrCondStats.put(attr, condStats);
    			}
    			int indx = condStats[0] == null ? 0 : 1;
    			condStats[indx] = new ConditionedFeatureStat(condAttrVal, totalCount, mean, variance);
    		}
    		//emit conditional stat
    		emitOutput( key,  context);
    	}	
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class ConditionedFeatureStat {
		private String condAttrVal;
		private int count;
		private double mean;
		private double variance;
		
		public ConditionedFeatureStat(String condAttrVal, int count, double mean, double variance) {
			super();
			this.condAttrVal = condAttrVal;
			this.count = count;
			this.mean = mean;
			this.variance = variance;
		}

		public String getCondAttrVal() {
			return condAttrVal;
		}

		public int getCount() {
			return count;
		}

		public double getMean() {
			return mean;
		}

		public double getVariance() {
			return variance;
		}
		
	}
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new FisherDiscriminant(), args);
        System.exit(exitCode);
	}

}
