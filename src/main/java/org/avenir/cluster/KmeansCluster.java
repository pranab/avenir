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
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.distance.AttributeDistanceSchema;
import org.chombo.distance.InterRecordDistance;
import org.chombo.stats.CategoricalHistogramStat;
import org.chombo.util.BasicUtils;
import org.chombo.util.GenericAttributeSchema;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * KMeans clustering at scale. Process multiple cluster groups in parallel
 * @author pranab
 *
 */
public class KmeansCluster extends Configured implements Tool {
	private static final String NULL = "null";
	
	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Kmeans cluster";
        job.setJobName(jobName);
        
        job.setJarByClass(KmeansCluster.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(KmeansCluster.ClusterMapper.class);
        job.setReducerClass(KmeansCluster.ClusterReducer.class);
                
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("kmc.num.reducer", -1);
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
        		clGrp.addCluster(centroid, movement, status, fieldDelimRegex);
        	}
        	
            for (String clGrpName :  clusterGroups.keySet()) {
            	ClusterGroup clGrp = clusterGroups.get(clGrpName);
            	clGrp.initialize();
            }
        }       
        
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex, -1);
            
            //all cluster groups
            for (String clGrpName :  clusterGroups.keySet()) {
            	ClusterGroup clGrp = clusterGroups.get(clGrpName);
            	if (clGrp.isActive()) {
            		ClusterGroup.Cluster nearest = clGrp.findClosestCluster(items, distanceFinder);
            		outKey.initialize();
            		outKey.add(clGrpName,  nearest.getCentroid());
            		
            		outVal.initialize();
            		outVal.add(value.toString(), nearest.getDistance());
                	context.write(outKey, outVal);
            	}
            }
            
        }       
	}	
	/**
	* @author pranab
  	*
  	*/
	public static class ClusterReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private Text outVal = new Text();
		private StringBuilder stBld =  new StringBuilder();;
		private String fieldDelim;
		private String clusterGroup;
		private String[] centroid;
		private String[] newCentroid;
        private int[] attrOrdinals;
        private GenericAttributeSchema schema;
        private AttributeDistanceSchema attrDistSchema;
        private InterRecordDistance distanceFinder;
        private Map<Integer, Double> numSums = new HashMap<Integer, Double>();
        private Map<Integer, CategoricalHistogramStat> catHist = new HashMap<Integer, CategoricalHistogramStat>();
        private int outputPrecision;
        private int count;
        private double sumDistSq;
        private double avError;
        private double movement;

		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration config = context.getConfiguration();
			fieldDelim = config.get("field.delim.out", ",");
			
 
        	//distance finder
        	schema = Utility.getGenericAttributeSchema(config, "kmc.schema.file.path");
        	attrDistSchema = Utility.getAttributeDistanceSchema(config, "kmc.attr.dist.schema.ath");
        	if (null == schema || null == attrDistSchema) {
        		throw new IllegalStateException("missing schema");
        	}
        	distanceFinder = new InterRecordDistance(schema, attrDistSchema, fieldDelim);
        	distanceFinder.withFacetedFields(attrOrdinals);
       	
        	//attributes
           	attrOrdinals = Utility.assertIntArrayConfigParam(config, "kmc.attr.odinals", Utility.configDelim, 
        			"missing attribute ordinals");
           	for (int attr : attrOrdinals) {
           		if (schema.areNumericalAttributes(attr)) {
           			numSums.put(attr, 0.0);
           		} else if (schema.areCategoricalAttributes(attr)) {
           			catHist.put(attr, new CategoricalHistogramStat());
           		} else {
           			throw new IllegalStateException("only numerical and categorical attribute allowed");
           		}
           	}
           	
            outputPrecision = config.getInt("nads.output.precision", 3);
		}
		
		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
     	throws IOException, InterruptedException {
			clusterGroup = key.getString(0);
			centroid = key.getString(1).split(fieldDelim, -1);
			int recSize = centroid.length;
			initialize();
			
			count = 0;
			sumDistSq = 0;
			for (Tuple val : values) {
				String[] rec = val.getString(0).split(fieldDelim, -1);
				for (int i = 0 ; i < rec.length; ++i) {
					Double sum = numSums.get(i);
					CategoricalHistogramStat hist = catHist.get(i);
					if (null != sum) {
						sum += Double.parseDouble(rec[i]);
						 numSums.put(i, sum);
					} else if (null != hist) {
						hist.add(rec[i]);
					} else {
						//attribute not included
					}
					double dist = val.getDouble(1);
					sumDistSq += dist * dist;
					++count;
				}
			}
			avError = sumDistSq / count;
			
			//numerical attr mean
			List<Integer> attrs = new ArrayList<Integer>(numSums.keySet());
			for (int attr : attrs) {
				double sum = numSums.get(attr);
				numSums.put(attr, sum/count);
			}
			
			//new centroid
			newCentroid = new String[recSize];
			for (int i = 0; i < recSize; ++i) {
				Double sum = numSums.get(i);
				CategoricalHistogramStat hist = catHist.get(i);
				if (null != sum) {
					newCentroid[i] = BasicUtils.formatDouble(sum,outputPrecision);
				} else if (null != hist) {
					newCentroid[i] = hist.getMode();
				} else {
					//attribute not included
					newCentroid[i] = NULL;
				}
			}
			
			//centroid, movement, status, sse, count
			movement = distanceFinder.findDistance(centroid, newCentroid);
			stBld.delete(0, stBld.length());
			stBld.append(clusterGroup).append(fieldDelim);
			for (String item : newCentroid) {
				stBld.append(item).append(fieldDelim);
			}
			stBld.append(BasicUtils.formatDouble(movement,outputPrecision)).append(fieldDelim).
				append(ClusterGroup.STATUS_ACTIVE).append(fieldDelim).
				append(BasicUtils.formatDouble(avError,outputPrecision)).append(fieldDelim).
				append(count);
			outVal.set(stBld.toString());
			context.write(NullWritable.get(), outVal);
		}		
		
		/**
		 * 
		 */
		private void initialize() {
			List<Integer> attrs = new ArrayList<Integer>(numSums.keySet());
			for (int attr : attrs) {
				numSums.put(attr, 0.0);
			}
			for (int attr : catHist.keySet()) {
				catHist.get(attr).intialize();
			}			
			
		}
	}
	
	

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new KmeansCluster(), args);
        System.exit(exitCode);
	}
	
}
