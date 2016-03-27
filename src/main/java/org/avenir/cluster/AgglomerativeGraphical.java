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

package org.avenir.cluster;

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
import org.avenir.util.EntityDistanceMapFileAccessor;
import org.chombo.util.Utility;

/**
 * @author pranab
 *
 */
public class AgglomerativeGraphical extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Agglomerative graph based clustering";
        job.setJobName(jobName);
        job.setJarByClass(AgglomerativeGraphical.class);
        Utility.setConfiguration(job.getConfiguration(), "avenir");

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
       
        job.setMapperClass(AgglomerativeGraphical.GraphMapper.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setNumReduceTasks(0);
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	
	/**
	 * Decision tree or random forest. For random forest, data is sampled in the first iteration
	 * @author pranab
	 *
	 */
	public static class GraphMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
		private String fieldDelimRegex;
		private String[] items;
		private List<EdgeWeightedCluster> clusters = new ArrayList<EdgeWeightedCluster>();
		private double minAvEdgeWeightThreshold;
		private String entityID;
		private EntityDistanceMapFileAccessor  distanceMapFileAccessor;
		private double avEdgeWeight;
		private double maxAvEdgeWeight;
		private Text outVal = new Text();
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
			Configuration config= context.getConfiguration();
            fieldDelimRegex = config.get("field.delim.regex", ",");
            minAvEdgeWeightThreshold = Utility.assertDoubleConfigParam(config, "agg.min.av.edge.weight.threshold", 
            		"missing min average edge weight");
            distanceMapFileAccessor = new EntityDistanceMapFileAccessor(config);
            distanceMapFileAccessor.initReader("agg.map.file.dir.path.param");
        }    

	   	/* (non-Javadoc)
	   	 * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
	   	 */
	   	@Override
	   	protected void cleanup(Context context)  throws IOException, InterruptedException {
	           for (EdgeWeightedCluster cluster :  clusters) {
	        	   	outVal.set(cluster.toString());
					context.write(NullWritable.get(), outVal);
	           }   
	   	}
	   	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
            entityID = items[0];
            
            maxAvEdgeWeight = Double.MIN_VALUE;
            EdgeWeightedCluster selCluster = null;
            for (EdgeWeightedCluster cluster :  clusters) {
            	avEdgeWeight = cluster.tryMembership(entityID, distanceMapFileAccessor);
            	if (avEdgeWeight > maxAvEdgeWeight) {
            		maxAvEdgeWeight = avEdgeWeight;
            		selCluster = cluster;
            	}
            }
            
            if(maxAvEdgeWeight > minAvEdgeWeightThreshold) {
            	//add to the best cluster found
            	selCluster.add(entityID, maxAvEdgeWeight);
            } else {
            	//create new cluster
            	clusters.add(new EdgeWeightedCluster());
            }
        }
 	}	

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new AgglomerativeGraphical(), args);
        System.exit(exitCode);
	}

}
