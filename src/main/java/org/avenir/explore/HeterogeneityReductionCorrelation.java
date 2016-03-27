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

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * @author pranab
 * Heterogeneity reduction based correlation for categorical data
 *
 */
public class HeterogeneityReductionCorrelation extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Categorical data correlation with Heterogeneity Reduction";
        job.setJobName(jobName);
        
        job.setJarByClass(HeterogeneityReductionCorrelation.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(CategoricalCorrelation.CorrelationMapper.class);
        job.setReducerClass(HeterogeneityReductionCorrelation.CorrelationReducer.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Text.class);

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
	public static class CorrelationReducer extends CategoricalCorrelation.CorrelationReducer {
		private String heterogeneityAlgorithm;
		
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);
			heterogeneityAlgorithm = context.getConfiguration().get("hrc.heterogeneity.algorithm", "gini");			
		}
		
		@Override
		protected double getCorrelationStat() {
			double stat = 0;
			if (heterogeneityAlgorithm.equals("gini")) {
				stat = contMat.concentrationCoeff();
			} else {
				stat = contMat.uncertaintyCoeff();
			}
			return stat;
		}
		
	}

	
}
