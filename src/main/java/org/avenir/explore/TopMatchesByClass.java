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
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.util.BasicUtils;
import org.chombo.util.SecondarySort;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Top match map reduce based on distance with neighbors and partitioned by 
 * class attribute
 * @author pranab
 *
 */
public class TopMatchesByClass extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Top n matches by class attribute  MR";
        job.setJobName(jobName);
        
        job.setJarByClass(TopMatchesByClass.class);
        
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        job.setMapperClass(TopMatchesByClass.TopMatchesMapper.class);
        job.setReducerClass(TopMatchesByClass.TopMatchesReducer.class);
        job.setCombinerClass(TopMatchesByClass.TopMatchesCombiner.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        
        job.setGroupingComparatorClass(SecondarySort.TuplePairGroupComprator.class);
        job.setPartitionerClass(SecondarySort.TuplePairPartitioner.class);

        Utility.setConfiguration(job.getConfiguration());
        int numReducer = job.getConfiguration().getInt("tmc.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);
        
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class TopMatchesMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private String srcEntityId;
		private String trgEntityId;
		private int rank;
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private String fieldDelim;
        private int recLength = -1;
        private int srcRecBeg;
        private int srcRecEnd;
        private int trgRecBeg;
        private int trgRecEnd;
        private int classAttrOrd;
        private String srcClassAttr;
        private String trgClassAttr;
        private String filterClassVal;
        private boolean doEmit;
        private boolean idInInput;
        private boolean includeRecInOutput;
        private String srcRec;
        private String trgRec;
        private int idOrd;

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
			Configuration config = context.getConfiguration();
           	fieldDelim = config.get("field.delim", ",");
            fieldDelimRegex = config.get("field.delim.regex", ",");
        	classAttrOrd = Utility.assertIntConfigParam(config, "tmc.class.attr.ord", 
        			"missing class attribute ordinal");
        	filterClassVal = config.get("tmc.filer.class.value");
        	idInInput = config.getBoolean("tmc.id.in.input", true);
        	if (!idInInput) {
        		//ID needs to extracted from record
        		idOrd = Utility.assertIntConfigParam(config, "", "missing ID field ordinal");
        	}
        	includeRecInOutput = config.getBoolean("tmc.include.rec.in.output", true);
        }    

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            String[] items  =  value.toString().split(fieldDelimRegex);
            
            //record length
           	if (recLength == -1) {
           		//Optional 2 Ids, two record and rank
        		int addFieldCount = idInInput ? 3 : 1;
        		recLength = (items.length - addFieldCount) / 2;
           	}      
           	
        	//record boundaries
        	srcRecBeg = idInInput? 2 : 0;
        	srcRecEnd = srcRecBeg + recLength;
        	trgRecBeg = srcRecEnd;
        	trgRecEnd = trgRecBeg + recLength;
         	
            if (idInInput) {
            	//ID in the beginning
            	srcEntityId = items[0];
            	trgEntityId = items[1];
            } else {
            	//ID embedded in record
            	srcEntityId = items[idOrd];
            	trgEntityId = items[idOrd + recLength];
            }
            rank = Integer.parseInt(items[items.length - 1]);
            
        	srcClassAttr = items[srcRecBeg + classAttrOrd];
        	trgClassAttr = items[trgRecBeg + classAttrOrd];
        	
        	//extract records
			if (includeRecInOutput) {
            	srcRec = BasicUtils.join(items, srcRecBeg, srcRecEnd, fieldDelim);
            	trgRec = BasicUtils.join(items, trgRecBeg, trgRecEnd, fieldDelim);
			}
        	
        	doEmit = false;
        	//only for same classes
        	if (srcClassAttr.equals(trgClassAttr)) {
        		if (null != filterClassVal) {
        			//specific class
        			doEmit = srcClassAttr.equals(filterClassVal);
        		} else {
        			//any class
        			doEmit = true;
        		}
        	}
        	
        	if (doEmit) {
        		//first then second
				keyValInit();
        		outKey.add(srcEntityId, srcClassAttr);
        		if (includeRecInOutput) {
        			outKey.add( srcRec);
        		}
        		outKey.add( rank);
        		outVal.add(trgEntityId);         
           		if (includeRecInOutput) {
           			outVal.add(trgRec);
        		}
        		outVal.add( rank);         
        		context.write(outKey, outVal);
        		
        		//second then first
				keyValInit();
        		outKey.add(trgEntityId, srcClassAttr);
        		if (includeRecInOutput) {
        			outKey.add( trgRec);
        		}
        		outKey.add( rank);
        		outVal.add(srcEntityId);         
           		if (includeRecInOutput) {
           			outVal.add(srcRec);
        		}
        		outVal.add( rank);         
        		context.write(outKey, outVal);
        	} 
        }
        
        /**
         * 
         */
        private void keyValInit() {
            outKey.initialize();
            outVal.initialize();
        }
	}

    /**
     * @author pranab
     *
     */
    public static class TopMatchesCombiner extends Reducer<Tuple, Text, Tuple, Tuple> {
    	private boolean nearestByCount;
    	private int topMatchCount;
    	private int topMatchDistance;
		private int count;
		private int distance;
    
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
        	nearestByCount = conf.getBoolean("tmc.nearest.by.count", true);
        	if (nearestByCount) {
        		topMatchCount = conf.getInt("tmc.match.count", 10);
        	} else {
        		topMatchDistance = conf.getInt("tmc.match.distance", 200);
        	}
       }
        
       	/* (non-Javadoc)
    	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
    	 */
    	protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
    		count = 0;
        	for (Tuple value : values){
        		//count based neighbor
				if (nearestByCount) {
					context.write(key, value);
	        		if (++count == topMatchCount){
	        			break;
	        		}
				} else {
					//distance based neighbor
					distance = value.getInt(value.getSize() - 1);
					if (distance  <=  topMatchDistance ) {
						context.write(key, value);
					} else {
						break;
					}
				}
        	}
    	}
    }
    
    /**
     * @author pranab
     *
     */
    public static class TopMatchesReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
    	private boolean nearestByCount;
    	private boolean nearestByDistance;
    	private int topMatchCount;
    	private int topMatchDistance;
		private String srcEntityId;
		private String srcRec;
		private int count;
		private int distance;
		private Text outVal = new Text();
        private String fieldDelim;
        private boolean compactOutput;
        private List<String> targetList = new ArrayList<String>();
    	private String srcEntityClassAttr;
 		private StringBuilder stBld = new  StringBuilder();
        private boolean includeRecInOutput;
        private boolean includeClassInOutput;
           	
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
			Configuration config = context.getConfiguration();
           	fieldDelim = config.get("field.delim", ",");
        	nearestByCount = config.getBoolean("tmc.nearest.by.count", true);
        	nearestByDistance = config.getBoolean("tmc.nearest.by.distance", false);
        	if (nearestByCount) {
        		topMatchCount = Utility.assertIntConfigParam(config, "tmc.top.match.count", "missing top match max count");
        	} else {
        		topMatchDistance = Utility.assertIntConfigParam(config, "tmc.top.match.distance", "missing top match max distance");
        	}
        	
        	includeRecInOutput = config.getBoolean("tmc.include.rec.in.output", true);
        	compactOutput =  config.getBoolean("tmc.compact.output", false);     
        	includeClassInOutput = config.getBoolean("tmc.include.class.in.output", true);
        }
    	
    	/* (non-Javadoc)
    	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
    	 */
    	protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
    		srcEntityId  = key.getString(0);
    		srcEntityClassAttr = key.getString(1);
    		srcRec = includeRecInOutput ? key.getString(2) : null;
    		
    		count = 0;
    		boolean doEmitNeighbor = false;
    		targetList.clear();
        	for (Tuple value : values){
        		doEmitNeighbor = false;
        		
        		//count based neighbor
				if (nearestByCount) {
					doEmitNeighbor = true;
	        		if (++count >= topMatchCount){
	        			doEmitNeighbor = false;
	        		}
				} 
				
				//distance based neighbors
				if (nearestByDistance) {
					//distance based neighbor
					distance = value.getInt(value.getSize() - 1);
					if (distance  <=  topMatchDistance ) {
						if (!nearestByCount) {
							doEmitNeighbor = true;
						}
					} else {
						doEmitNeighbor = false;
					}
				}

				//collect neighbors
				if (doEmitNeighbor) {
					if (includeRecInOutput) {
						//record
						targetList.add(value.getString(1));
					} else {
						//entity Id
						targetList.add(value.getString(0));
					}
				}
        	}
        	
        	//emit in compact or expanded format
    		int numNeighbor = targetList.size();
        	if (compactOutput) {
        		//one record for source and all neighbors
        		if (numNeighbor > 0) {
 					stBld.delete(0, stBld.length());
 					if (includeRecInOutput) {
 						stBld.append(srcRec);						
 					} else {
 						stBld.append(srcEntityId);
 					}
 					if (includeClassInOutput) {
 						stBld.append(fieldDelim).append(srcEntityClassAttr);
 					}
					for (String target : targetList) {
						stBld.append(fieldDelim).append(target);
					}
					outVal.set(stBld.toString());
					context.write(NullWritable.get(), outVal);
        		}
        	} else {
        		//one record for each source and neighbor pair
				for (String target : targetList) {
					stBld.delete(0, stBld.length());
 					if (includeRecInOutput) {
 						stBld.append(srcRec);						
 					} else {
 						stBld.append(srcEntityId);
 					}
 					if (includeClassInOutput) {
 						stBld.append(fieldDelim).append(srcEntityClassAttr);
 					}
					stBld.append(fieldDelim).append(target);
					
					outVal.set(stBld.toString());
					context.write(NullWritable.get(), outVal);
				}
        	}
    	}
    }
	
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new TopMatchesByClass(), args);
        System.exit(exitCode);
	}
}
