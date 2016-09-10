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
import org.avenir.util.ClassBasedNeighborhood;
import org.chombo.util.Attribute;
import org.chombo.util.BasicUtils;
import org.chombo.util.GenericAttributeSchema;
import org.chombo.util.Pair;
import org.chombo.util.SecondarySort;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

public class ReliefFeatureRelevance extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Feature relevance with relief algorithm  MR";
        job.setJobName(jobName);
        
        job.setJarByClass(ReliefFeatureRelevance.class);
        
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        job.setMapperClass(ReliefFeatureRelevance.RelevanceMapper.class);
        job.setReducerClass(ReliefFeatureRelevance.RelevanceReducer.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        
        job.setGroupingComparatorClass(SecondarySort.TuplePairGroupComprator.class);
        job.setPartitionerClass(SecondarySort.TuplePairPartitioner.class);

        Utility.setConfiguration(job.getConfiguration());
        int numReducer = job.getConfiguration().getInt("rrf.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);
        
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class RelevanceMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private String fieldDelim;
        private List<ClassBasedNeighborhood> neighborhoods = new ArrayList<ClassBasedNeighborhood>();
        private int idOrd;
        private String id;
        private int subKey;

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
           	fieldDelim = conf.get("field.delim", ",");
            fieldDelimRegex = conf.get("field.delim.regex", ",");
            
            //initialize neighborhood
            List<String> lines = Utility.getFileLines(conf, "ffr.neighborhood.file.path");
            for (String line : lines) {
            	String[] record = line.split(fieldDelimRegex);
            	neighborhoods.add(new ClassBasedNeighborhood(record));
            }
            
            idOrd = Utility.assertIntConfigParam(conf, "ffr.id.ord", "missing id field ordinal");
        }   
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            String[] items  =  value.toString().split(fieldDelimRegex);
            id = items[idOrd];
            
            for (ClassBasedNeighborhood neighborhood : neighborhoods) {
                subKey = -1;
            	if (neighborhood.isSrcEntity(id)) {
            		subKey = 0;
            	} else if (neighborhood.isTrgEntity(id)) {
            		subKey = 1;
            	}
            	if (subKey >= 0) {
            		neighborhood.generateKey(outKey, subKey);
            		outVal.initialize();
            		outVal.add(subKey, value.toString());
         			context.write(outKey, outVal);
            	}
            }
        }        
	}
	
    /**
     * @author pranab
     *
     */
    public static class RelevanceReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private Text outVal = new Text();
    	private int[] attrOrdinals;
    	private GenericAttributeSchema schema;
    	private Map<Integer, Pair<Boolean, Double>> attrTypes = new HashMap<Integer, Pair<Boolean, Double>>();
    	private Map<Integer, Double> scores = new HashMap<Integer, Double>();
       	private String fieldDelim;
       	private String[] srcRec;
       	private String[] trgRec;
       	private double diff;
       	private boolean hit;
       	private int sampCount = 0;
    	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
			Configuration config = context.getConfiguration();
           	fieldDelim = config.get("field.delim", ",");
			attrOrdinals = Utility.assertIntArrayConfigParam(config, "ffr.attr.ordinals", Utility.DEF_FIELD_DELIM, 
					"missing attribute ordinals");
			schema = Utility.getGenericAttributeSchema(config, "ffr.attr.schema.file.path");
			
			for (int attrOrd : attrOrdinals) {
				Attribute attr = schema.findAttributeByOrdinal(attrOrd);
				 if (attr.isNumerical()) {
					double range = attr.getMax() - attr.getMin();
					attrTypes.put(attrOrd, new Pair<Boolean, Double>(true, range));
				} else if (attr.isCategorical()) {
					attrTypes.put(attrOrd, new Pair<Boolean, Double>(true, -1.0));
				} else {
					throw new IllegalArgumentException("only numerical or categorical attribute allowed");
				}
				scores.put(attrOrd, 0.0);
			}
        }
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        @Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			super.cleanup(context);	
			sampCount /= 2;
			for (int attrOrd : attrOrdinals) {
				double score = scores.get(attrOrd) / sampCount;
				outVal.set("" + attrOrd + fieldDelim + BasicUtils.formatDouble(score, 3));
				context.write(NullWritable.get(), outVal);
			}
        }   
        
        
    	/* (non-Javadoc)
    	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
    	 */
    	protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
    		++sampCount;
    		hit = key.getString(1).equals(key.getString(2));
    		
        	for (Tuple value : values){
        		if (value.getInt(0) == 0) {
        			//source entity
        			srcRec = value.getString(1).split(fieldDelim);
        		} else {
        			//target entities
        			trgRec = value.getString(1).split(fieldDelim);
        			for (int attrOrd : attrOrdinals) {
        				Pair<Boolean, Double> type = attrTypes.get(attrOrd);
        				if (type.getLeft()) {
        					//numerical
        					diff = Double.parseDouble(srcRec[attrOrd]) - Double.parseDouble(trgRec[attrOrd]);
        					diff = Math.abs(diff) / type.getRight();
        				} else {
        					//categorical
        					diff = srcRec[attrOrd].equals(trgRec[attrOrd]) ? 0 : 1;
        				}
        				if (hit) {
        					scores.put(attrOrd, scores.get(attrOrd) - diff);
        				} else {
        					scores.put(attrOrd, scores.get(attrOrd) + diff);
        				}
        			}
        			
        		}
        	}
    	}        
    }
    
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new ReliefFeatureRelevance(), args);
        System.exit(exitCode);
	}
    
}
