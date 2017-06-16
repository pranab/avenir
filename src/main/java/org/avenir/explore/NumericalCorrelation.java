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
import org.chombo.stats.NumericalAttrStatsManager;
import org.chombo.util.GenericAttributeSchema;
import org.chombo.util.Pair;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * @author pranab
 *
 */
public class NumericalCorrelation  extends Configured implements Tool {

	/* (non-Javadoc)
	 * @see org.apache.hadoop.util.Tool#run(java.lang.String[])
	 */
	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Cross correlation between  numerical attributes";
        job.setJobName(jobName);
        
        job.setJarByClass(NumericalCorrelation.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "chombo");
        job.setMapperClass(NumericalCorrelation.CorrelationMapper.class);
        job.setReducerClass(NumericalCorrelation.CorrelationReducer.class);
        job.setCombinerClass(NumericalCorrelation.CorrelationCombiner.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("nuc.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class CorrelationMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private int count = 1;
        private String[] items;
        private GenericAttributeSchema schema;
        private List<Pair<Integer, Integer>> attrPairs;
        private Map<Integer, Double> attrMean = new HashMap<Integer, Double>();
        private int firstAttr;
        private int secondAttr;
        private double first;
        private double second;
        private double sum;
        
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	attrPairs = Utility. assertIntPairListConfigParam(config, "nuc.attr.pairs",  Utility.configDelim, Utility.configSubFieldDelim, "");
        	
        	schema = Utility.getGenericAttributeSchema(config,  "nuc.schema.file.path");
        	if (null != schema) {
        		//validate
        		int[] attrs = new int[attrPairs.size() * 2];
        		int i = 0;
            	for (Pair<Integer, Integer> pair : attrPairs) {
            		attrs[i++] = pair.getLeft();
            		attrs[i++] = pair.getRight();
            	}
            	if (!schema.areNumericalAttributes(attrs)) {
            		throw new IllegalArgumentException("attributes should be numerical");
            	}
        	}
        	
        	//get mean values
        	NumericalAttrStatsManager statsManager = new NumericalAttrStatsManager( config, "nuc.stats.file.path", fieldDelimRegex);
        	for (Pair<Integer, Integer> pair : attrPairs) {
        		attrMean.put(pair.getLeft(), statsManager.getMean(pair.getLeft()));
        		attrMean.put(pair.getRight(), statsManager.getMean(pair.getRight()));
        	}
       }

        @Override
        protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
            items  =  value.toString().split(fieldDelimRegex);
        	for (Pair<Integer, Integer> pair : attrPairs) {
            	outKey.initialize();
            	outVal.initialize();
            	
            	firstAttr = pair.getLeft();
            	first = Double.parseDouble(items[firstAttr]);
            	secondAttr = pair.getRight();
            	second = Double.parseDouble(items[secondAttr]);
            	sum = (first -  attrMean.get(firstAttr)) * (second - attrMean.get(secondAttr));
            	
            	outKey.add(firstAttr, secondAttr);
            	outVal.add(sum, count); 
               	context.write(outKey, outVal);
        	}
        }
      
	}

	/**
	 * @author pranab
	 *
	 */
	public static class CorrelationCombiner extends Reducer<Tuple, Tuple, Tuple, Tuple> {
		private Tuple outVal = new Tuple();
		private double sum;
		private int totalCount;
		
        protected void reduce(Tuple  key, Iterable<Tuple> values, Context context)
        		throws IOException, InterruptedException {
    		sum = 0;
    		totalCount = 0;
    		int i = 0;
    		for (Tuple val : values) {
    			sum  += val.getDouble(0);
    			totalCount += val.getInt(1);
    		}
    		outVal.initialize();
    		outVal.add(sum, totalCount);
        	context.write(key, outVal);       	
        }		
	}	

	   /**
     * @author pranab
     *
     */
    public static class CorrelationReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
    	protected Text outVal = new Text();
		protected StringBuilder stBld =  new StringBuilder();;
		protected String fieldDelim;
		private double sum;
		private int totalCount;
        private Map<Integer, Double> attrStdDev = new HashMap<Integer, Double>();
        private double corr;
        private List<Pair<Integer, Integer>> attrPairs;

		/* (non-Javadoc)
		 * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
		 */
		protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelim = config.get("field.delim.out", ",");
        	NumericalAttrStatsManager statsManager = new NumericalAttrStatsManager(config, "nuc.stats.file.path", fieldDelim);
        	attrPairs = Utility. assertIntPairListConfigParam(config, "nuc.attr.pairs",  Utility.configDelim, Utility.configSubFieldDelim, "");
        	for (Pair<Integer, Integer> pair : attrPairs) {
        		attrStdDev.put(pair.getLeft(), statsManager.getStdDev(pair.getLeft()));
        		attrStdDev.put(pair.getRight(), statsManager.getStdDev(pair.getRight()));
        	}
     }
		
    	/* (non-Javadoc)
    	 * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
    	 */
    	protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
    		sum = 0;
    		totalCount = 0;
    		int i = 0;
    		for (Tuple val : values) {
    			sum  += val.getDouble(0);
    			totalCount += val.getInt(1);
    		}
    		sum /= totalCount;
    		corr = sum / (attrStdDev.get(key.getInt(0)) *  attrStdDev.get(key.getInt(1)));
    		outVal.set("" + key.getInt(0) + fieldDelim +key.getInt(1) + fieldDelim + corr );
			context.write(NullWritable.get(), outVal);
    	}
    }
    
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new NumericalCorrelation(), args);
        System.exit(exitCode);
	}
    
}
