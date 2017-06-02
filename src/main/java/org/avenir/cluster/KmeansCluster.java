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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.chombo.distance.AttributeDistanceSchema;
import org.chombo.distance.InterRecordDistance;
import org.chombo.mr.NumericalAttrDistrStats;
import org.chombo.util.BasicUtils;
import org.chombo.util.GenericAttributeSchema;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * @author pranab
 *
 */
public class KmeansCluster extends Configured implements Tool {
	private static final String STATUS_ACTIVE = "active";
	private static final String STATUS_STOPPED = "stopped";
	
	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Kmeans cluster";
        job.setJobName(jobName);
        
        job.setJarByClass(KmeansCluster.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "chombo");
        job.setMapperClass(KmeansCluster.ClusterMapper.class);
        job.setCombinerClass(NumericalAttrDistrStats.StatsCombiner.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("nads.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class ClusterMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private String[] items;
        private int[] attrOrdinals;
        private GenericAttributeSchema schema;
        private AttributeDistanceSchema attrDistSchema;
        private InterRecordDistance distanceFinder;
        private Map<String, ClusterGroup> clusterGroups = new HashMap<String, ClusterGroup>();
       
  
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	attrOrdinals = Utility.assertIntArrayConfigParam(config, "kmc.attr.odinals", Utility.configDelim, 
        			"missing attribute ordinals");
        	
        	//distance finder
        	schema = Utility.getGenericAttributeSchema(config, "kmc.schema.file.path");
        	attrDistSchema = Utility.getAttributeDistanceSchema(config, "kmc.attr.dist.schema.ath");
        	if (null == schema || null == attrDistSchema) {
        		throw new IllegalStateException("missing schema");
        	}
        	distanceFinder = new InterRecordDistance(schema, attrDistSchema, fieldDelimRegex);
        	distanceFinder.withFacetedFields(attrOrdinals);

        	//cluster initialization
        	double movementThreshold = Utility.assertDoubleConfigParam(config, "kmc.movement.threshold", "missing movement ");
        	List<String> lines = Utility. assertFileLines(config, "kmc.cluster.file.path",  "missing cluster  file");
        	int numAttributes = schema.getAttributeCount();
        	for (String line : lines) {
        		//cluster group
        		String[] items = BasicUtils.splitOnFirstOccurence(line,  fieldDelimRegex, true);
        		String clustGroup = items[0];
        		
        		//centoiid 
        		String centroid = items[1];
        		int pos =  BasicUtils.findOccurencePosition(centroid,fieldDelimRegex,numAttributes,true);
        		items = BasicUtils.splitOnPosition(centroid, pos, fieldDelimRegex.length(),  true);
        		centroid = items[0];
        		
        		//remaining
        		items = items[1].split(fieldDelimRegex, -1);
        		double movement = Double.parseDouble(items[0]);
        		String status = items[1];
        		
        		ClusterGroup clGrp = clusterGroups.get(clustGroup);
        		if (null == clGrp) {
        			clGrp = new ClusterGroup(movementThreshold);
        			clusterGroups.put(clustGroup, clGrp);
        		}
        		clGrp.addCluster(centroid, movement, status);
        	}
        }       
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex, -1);
            
            
        }       
	}	
	
	/**
	 * @author pranab
	 *
	 */
	private static class ClusterGroup {
		private List<Cluster> clusters = new ArrayList<Cluster>();
		private double movementThreshold;
		
		/**
		 * @param movementThreshold
		 */
		public ClusterGroup(double movementThreshold) {
			super();
			this.movementThreshold = movementThreshold;
		}

		/**
		 * @param centroid
		 * @param movement
		 * @param status
		 */
		public void addCluster(String centroid,  double movement, String status) {
			status = movement < movementThreshold ?  STATUS_STOPPED  :  status;
			clusters.add( new Cluster(centroid,  movement, status));
		}
		
		/**
		 * @return
		 */
		public boolean isActive() {
			boolean isActive = false;
			for( Cluster cluster : clusters ) {
				if (cluster.status.equals(STATUS_ACTIVE)) {
					isActive = true;
					break;
				}
			}
			return isActive;
		}
	}
	
	/**
	 * @author pranab
	 *
	 */
	private static class Cluster {
		private String centroid;
		private double movement;
		private String status; 
		
		public Cluster(String centroid,  double movement, String status) {
			this.centroid = centroid;
            this.movement = movement;
            this.status = status;
		}

		public String getCentroid() {
			return centroid;
		}

		public double getMovement() {
			return movement;
		}

		public String getStatus() {
			return status;
		}

	}

}
