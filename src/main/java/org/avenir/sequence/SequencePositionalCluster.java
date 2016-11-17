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

package org.avenir.sequence;

import java.io.IOException;
import java.util.Arrays;
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
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.chombo.util.Utility;
import org.hoidla.query.Criteria;
import org.hoidla.util.ExplicitlyTimetStampedValue;
import org.hoidla.window.EventLocality;
import org.hoidla.window.TimeBoundEventLocalityAnalyzer;

/**
 * For data points within a window that meet certain criteria, clustering is detected based on the
 * position of those data points within the window
 * @author pranab
 *
 */
public class SequencePositionalCluster  extends Configured implements Tool {
	private static final String configDelim = ",";
	private static final String configSubDelim = ":";

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Clustering based postion of data points in sequence meeting certain criteria ";
        job.setJobName(jobName);
        
        job.setJarByClass(SequencePositionalCluster.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(SequencePositionalCluster.ClusterMapper.class);
        
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        job.setNumReduceTasks(0);

        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class ClusterMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
		private String fieldDelimRegex;
		private String fieldDelim;
		protected Text outVal = new Text();
		private String[] items;
		private int skipFieldCount;
        private int[] idOrdinals;
        private long windowTimeSpan;
        private long timeStep;
        private int quantFieldOrdinal;
        private int seqNumFieldOrdinal;
        private TimeBoundEventLocalityAnalyzer window;
        private long timeStamp;
        private double quantValue;
        private ExplicitlyTimetStampedValue<Double> windowData;
        private double score;
        private double scoreThreshold;
        private Criteria criteria;
        private double[] operandValues;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
    		fieldDelim = conf.get("field.delim.out", ",");

    		windowTimeSpan = Utility.assertIntConfigParam(conf, "window.time.span", "wondow time span must be specified");
        	timeStep = Utility.assertIntConfigParam(conf, "processing.time.step", "missing window processing time step");
        	quantFieldOrdinal = Utility.assertIntConfigParam(conf, "quant.field.ordinal",  "missing quantity field ordinal");
        	seqNumFieldOrdinal = Utility.assertIntConfigParam(conf, "seq.num..field.ordinal",  "missing sequence field ordinal ");
        	
        	//window object
        	boolean isWeightedStrategy = Utility.assertBooleanConfigParam(conf, "wejghter.strategy", 
        			"weighted strategy flag should be provided");
        	EventLocality.Context strategyContext;
        	if (isWeightedStrategy) {
        		Map<String, Double> weightedStrategies = Utility.assertDoubleMapConfigParam(conf, "weighted.strategies", configDelim, 
        				configSubDelim, "missing weighted starategy configuration");
        		strategyContext  = new EventLocality.Context(weightedStrategies);
        	} else {
        		int minOccurence = Utility.assertIntConfigParam( conf, "min.occurence",  
        				"missing min occurence parameter");
        		int maxIntervalAverage = Utility.assertIntConfigParam( conf, "max.interval.average", 
        				"missing max interval average parameter");
        		int maxIntervalMax = Utility.assertIntConfigParam( conf, "max.interval.max",  
        				"missing max interval maximum  parameter");
        		List<String> preferredStrategies = Arrays.asList(Utility.assertStringArrayConfigParam(conf, "preferred.strategies", 
        				configDelim, "missing preferred strategies list"));
        		boolean anyCond = Utility.assertBooleanConfigParam(conf, "any.cond", "missing any condition flag");
        		long minRangeLength = conf.getLong("min.range.length", 0);
        		strategyContext  = new EventLocality.Context(minOccurence, (long)maxIntervalAverage, (long)maxIntervalMax, 
        				(long)minRangeLength, preferredStrategies,  anyCond);
        	}
        	long minEventTimeInterval = conf.getLong("min.event.time.interval", 100);
        	double scoreThreshold = Utility.assertDoubleConfigParam(conf, "score.threshold", "missing score threshold");
        	window = new TimeBoundEventLocalityAnalyzer(windowTimeSpan, timeStep, minEventTimeInterval, scoreThreshold, strategyContext);
        	scoreThreshold = Utility.assertDoubleConfigParam(conf, "score.threshold",  "missing score threhold parameter");
        	
        	String condExpression = Utility.assertStringConfigParam(conf, "cond.expression", "mission conditional expression");
        	criteria = Criteria.createCriteriaFromExpression(condExpression);
        	operandValues = new double[criteria.getNumPredicates()];
        }

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	quantValue = Double.parseDouble(items[quantFieldOrdinal]);
        	timeStamp = Long.parseLong(items[seqNumFieldOrdinal]);
        	
        	//add to window
        	for (int i = 0; i < operandValues.length; ++i) {
        		operandValues[i] = quantValue;
        	}
        	windowData = new ExplicitlyTimetStampedValue<Double>(quantValue, timeStamp, isConditionMet());
        	window.add(windowData);
        	score = window.getScore();
        	if (score > scoreThreshold) {
        		outVal.set(items[seqNumFieldOrdinal] + fieldDelim +  items[quantFieldOrdinal] + fieldDelim + score);
    			context.write(NullWritable.get(), outVal);
        	}
        }	
        
        private boolean isConditionMet() {
         	return criteria.evaluate(operandValues);
        }
	}	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new SequencePositionalCluster(), args);
        System.exit(exitCode);
	}
	
}
