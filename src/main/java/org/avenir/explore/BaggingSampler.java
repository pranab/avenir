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
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.chombo.util.Utility;

/**
 * Implements bagging sampler. Applies bagging on aper batch basis, since we can
 * not sample the whole data set
 * @author pranab
 *
 */
public class BaggingSampler extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Bagging sampler ";
        job.setJobName(jobName);
        
        job.setJarByClass(BaggingSampler.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(BaggingSampler.BaggingMapper.class);

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
	public static class BaggingMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
		private int batchSize;
		private List<Text> batch = new ArrayList<Text>();
		private int rowCount = 0;
	    private static final Logger LOG = Logger.getLogger(BaggingMapper.class);
		
		
       /* (non-Javadoc)
        * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
        */
	    protected void setup(Context context) throws IOException, InterruptedException {
	    	Configuration conf = context.getConfiguration();
	    	if (conf.getBoolean("debug.on", false)) {
	    		LOG.setLevel(Level.DEBUG);
	    	}
	    	batchSize = conf.getInt("bas.batch.size", 10000);
	    }	
       
       /* (non-Javadoc)
        * @see org.apache.hadoop.mapreduce.Mapper#cleanup(org.apache.hadoop.mapreduce.Mapper.Context)
        */
	    protected void cleanup(Context context)  throws IOException, InterruptedException {
	    	emitBatch(batch.size(),  context);
	    }
       
	    @Override
	    protected void map(LongWritable key, Text value, Context context)
           throws IOException, InterruptedException {
    	   batch.add(new Text(value));
    	   if (++rowCount == batchSize) {
    		   emitBatch(batchSize,  context);
    		   batch.clear();
    		   rowCount = 0;
    	   }
	    }
       
       /**
        * @param batchSize
        * @param context
        * @throws IOException
        * @throws InterruptedException
        */
	    private void emitBatch(int batchSize, Context context) throws IOException, InterruptedException {
	    	for (int i= 0; i < batchSize; ++i) {
	    		int sel = (int)(Math.random() * batchSize);
	    		context.write(NullWritable.get(),batch.get(sel));
	    	}
	    }
	}
	
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new BaggingSampler(), args);
        System.exit(exitCode);
	}
	
}
