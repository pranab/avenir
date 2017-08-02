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
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.avenir.util.RuleExpression;
import org.chombo.util.AttributeFilter;
import org.chombo.util.BasicUtils;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * @author pranab
 *
 */
public class RuleEvaluator extends Configured implements Tool {
	private static String CONF_ENTROPY = "confEntropy";
	private static String CONF_ACCURACY = "confAccuracy";
	

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Rule evaluator  MR";
        job.setJobName(jobName);
        
        job.setJarByClass(RuleEvaluator.class);

        FileInputFormat.addInputPaths(job, args[0]);
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration());
        
        job.setMapperClass(RuleEvaluator.EvaluatorMapper.class);
        job.setReducerClass(RuleEvaluator.EvaluatorReducer.class);
        job.setCombinerClass(RuleEvaluator.EvaluatorCombiner.class);
        
        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        int numReducer = job.getConfiguration().getInt("rue.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);
        
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class EvaluatorMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private String[] items;
        private Map<String, RuleExpression> ruleExpressions = new HashMap<String, RuleExpression>();
        private int classAttrOrdinal;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("field.delim.regex", ",");

        	String condDelim = config.get("rue.cond.delim");
        	if (null != condDelim) {
        		AttributeFilter.setConjunctSeparator(condDelim);
        	}
        	String[] ruleNames = Utility.assertStringArrayConfigParam(config, "rue.rule.names", Utility.configDelim, 
        			"missing rule list");
        	for (String ruleName : ruleNames) {
        		String key = "rue.rule." + ruleName;
        		String rule = Utility.assertStringConfigParam(config, key, "missing rule definition");
        		RuleExpression ruleExp = RuleExpression.createRule(rule);
        		ruleExpressions.put(ruleName, ruleExp);
        	}
        	classAttrOrdinal = Utility.assertIntConfigParam(config, "rue.class.attr.ord", 
        			"missing class attribute ordinal");
        	
        	
        }	
	    
        @Override
	    protected void map(LongWritable key, Text value, Context context)
	    	throws IOException, InterruptedException {
        	items  = value.toString().split(fieldDelimRegex, -1);
        	
        	//all rules
        	for (String ruleName : ruleExpressions.keySet()) {
        		RuleExpression ruleExp = ruleExpressions.get(ruleName);
        		if (ruleExp.evaluate(items)) {
        			outKey.initialize();
        			outKey.add(ruleName);
        			
        			outVal.initialize();
        			outVal.add(items[classAttrOrdinal], 1);
                	context.write(outKey, outVal);
        		}
        	}
        }
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class EvaluatorCombiner extends Reducer<Tuple, Tuple, Tuple, Tuple> {
		private Tuple outVal = new Tuple();
		private Map<String, Integer> classCounts = new HashMap<String, Integer>();
        private String classVal;
        private int classCount;
		
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple  key, Iterable<Tuple> values, Context context)
        		throws IOException, InterruptedException {
        	classCounts.clear();
        	for (Tuple value : values){
        		int i = 0;
        		classVal = value.getString(i++);
        		classCount = value.getInt(i++);
        		addToConsCount();
        		if (value.getSize() > 2) {
        			classVal = value.getString(i++);
        			classCount = value.getInt(i++);
            		addToConsCount();
        		}
        	}
        	outVal.initialize();
        	for (String clVal : classCounts.keySet()) {
        		outVal.add(clVal, classCounts.get(clVal));
        	}
        	context.write(key, outVal);       	
        }
        
        /**
         * 
         */
        private void addToConsCount() {
        	Integer count = classCounts.get(classVal);
        	if (count == null) {
        		classCounts.put(classVal, classCount);
        	} else {
        		classCounts.put(classVal, count + classCount);
        	}
        }
        
	}
	
    /**
     * @author pranab
     *
     */
    public static class EvaluatorReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private Text outVal = new Text();
		private String fieldDelimOut;
		private Map<String, Integer> classCounts = new HashMap<String, Integer>();
		private Map<String, String> ruleConsequents = new HashMap<String, String>();
        private String confStrategy;
        private int dataSize;
        private int totalCount;
        private String classVal;
        private int classCount;
        private double confidence;
        private double support;
        private String[] classValues;

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimOut = config.get("field.delim", ",");
        	confStrategy = Utility.assertStringConfigParam(config, "rue.conf.strategy",
        		"missing confidence strategy list");
        	dataSize = Utility.assertIntConfigParam(config, "rue.data.size", "missing data size");
        	String[] ruleNames = Utility.assertStringArrayConfigParam(config, "rue.rule.names", Utility.configDelim, 
        			"missing rule list");
        	for (String ruleName : ruleNames) {
        		String key = "rue.rule." + ruleName;
        		String rule = Utility.assertStringConfigParam(config, key, "missing rule definition");
        		ruleConsequents.put(ruleName, RuleExpression.extractConsequent(rule));
        	}
        	classValues = Utility.assertStringArrayConfigParam(config, "rue.class.values", Utility.configDelim, 
        			"missing class values");
          	
        }
    
		/* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
        	classCounts.clear();
        	totalCount = 0;
        	for (Tuple value : values){
        		int i = 0;
        		classVal = value.getString(i++);
        		classCount = value.getInt(i++);
        		addToConsCount();
        		totalCount += classCount;
        		if (value.getSize() > 2) {
        			classVal = value.getString(i++);
        			classCount = value.getInt(i++);
            		addToConsCount();
            		totalCount += classCount;
        		}
        	}
        	
        	//confidence and support
       		String ruleName = key.getString(0);
    		classVal = ruleConsequents.get(ruleName);
    		if (confStrategy.equals(CONF_ACCURACY)) {
    			confidence = ((double)getConsCount(classVal)) / totalCount;
        	} else if (confStrategy.equals(CONF_ENTROPY)) {
        		String otherClassVal = getOtherClassValue(classVal);
        		double prThisClass =  ((double)getConsCount(classVal)) / totalCount;
        		double prOtherClass =  ((double)getConsCount(otherClassVal)) / totalCount;
        		confidence = (prThisClass * Math.log(prThisClass) + prOtherClass * Math.log(prOtherClass)) / Math.log(2);
        		confidence += 1.0;
        	} else {
        		throw new IllegalStateException("invalid confidence strategy");
        	}
    		support = ((double)totalCount) / dataSize;
    		
    		outVal.set(ruleName + fieldDelimOut + BasicUtils.formatDouble(confidence, 3) + 
    				fieldDelimOut + BasicUtils.formatDouble(support, 3));
        	context.write(NullWritable.get(), outVal);
        }
        
        /**
         * @param clVal
         * @return
         */
        private int getConsCount(String clVal) {
        	Integer count = classCounts.get(clVal);
        	count = count != null ?  count : 0;
        	return count;
        }
        
        /**
         * 
         */
        private void addToConsCount() {
        	Integer count = classCounts.get(classVal);
        	if (count == null) {
        		classCounts.put(classVal, classCount);
        	} else {
        		classCounts.put(classVal, count + classCount);
        	}
        }
        
        /**
         * @param classVal
         * @return
         */
        private String getOtherClassValue(String classVal) {
        	String otherClassVal = null;
        	for (int i = 0; i < classValues.length; ++i) {
        		if (classValues[i].equals(classVal)) {
        			otherClassVal = classValues[i ^ 1];
        			break;
        		}
        	}
        	return otherClassVal;
        }
    }

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new RuleEvaluator(), args);
        System.exit(exitCode);
	}
    
}
