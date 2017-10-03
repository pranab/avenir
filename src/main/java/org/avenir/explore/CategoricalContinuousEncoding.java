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
import java.util.HashMap;
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
import org.avenir.util.ClassAttributeCounter;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Continuous encoding of categorical attributes. Particularly useful for high cardinality
 * categorical attributes
 * @author pranab
 *
 */
public class CategoricalContinuousEncoding extends Configured implements Tool {
	private static String ANY_VALUE = "*";
	private static String SUPERVISED_RATIO = "supervisedRatio";
	private static String WEIGHT_OF_EVIDENCE = "weightOfEvidence";
	

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Categorical attribute continuous encoding  MR";
        job.setJobName(jobName);
        
        job.setJarByClass(CategoricalContinuousEncoding.class);

        FileInputFormat.addInputPaths(job, args[0]);
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration());
        
        job.setMapperClass(CategoricalContinuousEncoding.EncoderMapper.class);
        job.setCombinerClass(CategoricalContinuousEncoding.EncoderCombiner.class);
        job.setReducerClass(CategoricalContinuousEncoding.EncoderReducer.class);

        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("coe.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);
        
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class EncoderMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private int[] attrOrdinals;
        private String encodingStrategy;
        private String[] items;
        private int classAttrOrd;
        private String posClassAttrValue;
        private String classAttrValue;
        private boolean isWeightOfEvidence;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	
        	attrOrdinals = Utility.assertIntArrayConfigParam(config, "coe.cat.attribute.ordinals", Utility.DEF_FIELD_DELIM, 
        			"missing categorical attribute ordinals");
        	encodingStrategy = Utility.assertStringConfigParam(config, "coe.encoding.strategy", "missing encoding strategy");
        	classAttrOrd = Utility.assertIntConfigParam(config, "coe.class.attr.ordinal", "missing class atrribute ordinal");
        	posClassAttrValue = Utility.assertStringConfigParam(config, "coe.pos.class.attr.value", "missing positive class attribute value");
        	isWeightOfEvidence = encodingStrategy.equals(WEIGHT_OF_EVIDENCE);
        }	
	    
        @Override
	    protected void map(LongWritable key, Text value, Context context)
	    		throws IOException, InterruptedException {
        	items  = value.toString().split(fieldDelimRegex, -1);
        	classAttrValue = items[classAttrOrd];
        	boolean isPositive = classAttrValue.equals(posClassAttrValue);
        	
        	//all attributes
        	for (int ord : attrOrdinals) {
        		outKey.initialize();
        		outKey.add(ord, items[ord]);
        		populateValue(isPositive);
        		context.write(outKey, outVal);
        	}
	        
        	//total class attribute count
    		if (isWeightOfEvidence) {
        		outKey.initialize();
        		outKey.add(-1, ANY_VALUE);
        		populateValue(isPositive);
        		context.write(outKey, outVal);
    		}
        }
        
        /**
         * @param isPositive
         */
        private void populateValue(boolean isPositive) {
    		outVal.initialize();
    		if (isPositive) {
    			outVal.add(1, 0);
    		} else {
    			outVal.add(0, 1);
    		}
        }
        
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class EncoderCombiner extends Reducer<Tuple, Tuple, Tuple, Tuple> {
		private Tuple outVal = new Tuple();
		private ClassAttributeCounter classAttrCounter = new ClassAttributeCounter();
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<Tuple> values, Context context)
        		throws IOException, InterruptedException {
        	classAttrCounter.initialize();
        	for (Tuple value : values){
        		classAttrCounter.add(value.getInt(0), value.getInt(1));
        	}
        	outVal.initialize();
        	outVal.add(classAttrCounter.getPosCount(), classAttrCounter.getNegCount());
    		context.write(key, outVal);
        }		
	}
	
    /**
     * @author pranab
     *
     */
    public static class EncoderReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private Text outVal = new Text();
		private String fieldDelimOut;
		private StringBuilder stBld = new  StringBuilder();
		private Map<Tuple, ClassAttributeCounter> classAttrCounter = new HashMap<Tuple, ClassAttributeCounter>();
        private boolean isWeightOfEvidence;
        private int scale;

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimOut = config.get("field.delim", ",");
        	String encodingStrategy = Utility.assertStringConfigParam(config, "coe.encoding.strategy", 
        			"missing encoding strategy");
        	isWeightOfEvidence = encodingStrategy.equals(WEIGHT_OF_EVIDENCE);
        	scale = Utility.assertIntConfigParam(config, "coe.output.scale", "missing output scale");
        }
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#cleanup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        @Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			super.cleanup(context);	
			ClassAttributeCounter allCounter = null;
			if (isWeightOfEvidence) {
				Tuple allKey = new Tuple();
				allKey.add(-1, ANY_VALUE);
				allCounter = classAttrCounter.remove(allKey);
			}
			int value = 0;
			for (Tuple key : classAttrCounter.keySet()) {
				stBld.delete(0, stBld.length());
				ClassAttributeCounter counter = classAttrCounter.get(key);
				if (isWeightOfEvidence) {
					double woe = ((double)counter.getPosCount()) / allCounter.getPosCount();
					int negCount = counter.getNegCount() == 0 ?  1 :   counter.getNegCount();
					woe /= ((double)negCount) / allCounter.getNegCount();
					woe = Math.log(woe);
					value = (int)(woe * scale);
				} else {
					value = (counter.getPosCount() * scale) / counter.getTotalCount();
				}
				stBld.append(key.get(0)).append(fieldDelimOut).append(key.get(1)).append(fieldDelimOut).append(value);
				outVal.set(stBld.toString());
				context.write(NullWritable.get(), outVal);
			}
		}

		/* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
        	ClassAttributeCounter counter = classAttrCounter.get(key);
        	if (null == counter) {
        		counter = new ClassAttributeCounter();
        		classAttrCounter.put(key.createClone(), counter);
        	}
        	
        	//update counts
        	for (Tuple value : values){
        		counter.add(value.getInt(0), value.getInt(1));
        	}
        }
    }
    
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new CategoricalContinuousEncoding(), args);
        System.exit(exitCode);
	}
    
}
