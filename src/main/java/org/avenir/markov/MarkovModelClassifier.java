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

package org.avenir.markov;

import java.io.IOException;
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
 * Markov model based classifier. 
 * @author pranab
 *
 */
public class MarkovModelClassifier extends Configured implements Tool {
	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Markov model based classifier";
        job.setJobName(jobName);
        
        job.setJarByClass(MarkovModelClassifier.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration(), "avenir");
        job.setMapperClass(MarkovModelClassifier.ClassifierMapper.class);
        
        job.setMapOutputKeyClass(NullWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setNumReduceTasks(0);
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class ClassifierMapper extends Mapper<LongWritable, Text, NullWritable, Text> {
		private String fieldDelimRegex;
		private String fieldDelim;
		private String[] items;
		private int skipFieldCount;
		private Text outVal  = new Text();
		private boolean isClassLabelBased;
		private MarkovModel model;
		private String frState;
		private String toState;
		private String[] classLabels;
		private double logOdds;
		private int idFieldOrd;
		private String predClass;
		private boolean inValidationMode;
		private StringBuilder stBld =  new StringBuilder();
		private int classLabelFieldOrd = -1;
		private int transProbScale;
		private double logOddsThreshold = 0;
        private static final Logger LOG = Logger.getLogger(ClassifierMapper.class);

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration conf = context.getConfiguration();
            if (conf.getBoolean("debug.on", false)) {
            	LOG.setLevel(Level.DEBUG);
            }
        	fieldDelimRegex = conf.get("field.delim.regex", ",");
        	fieldDelim = conf.get("field.delim.out", ",");
            skipFieldCount = conf.getInt("mmc.skip.field.count", 1);
            idFieldOrd = conf.getInt("mmc.id.field.ord", 0);
            isClassLabelBased = conf.getBoolean("mmc.class.label.based.model", false);
            inValidationMode = conf.getBoolean("mmc.validation.mode", false);
            if (inValidationMode) {
            	++skipFieldCount;
            	classLabelFieldOrd = conf.getInt("mmc.class.label.field.ord", -1);
            	if (classLabelFieldOrd < 0) {
            		throw new IllegalArgumentException("In validation mode actual class labels must be provided");
            	}
            }
            
        	List<String> lines = Utility.getFileLines(conf, "mmc.mm.model.path");
        	model = new MarkovModel(lines,  isClassLabelBased);
        	classLabels = conf.get("mmc.class.labels").split(",");
        	transProbScale = conf.getInt("mmc.trans.prob.scale", 1000);

        	if (null != conf.get("mmc.log.odds.threshold")) {
        		logOddsThreshold = Double.parseDouble(conf.get("mmc.log.odds.threshold"));
        	} 
        	LOG.debug("logOddsThreshold:" + logOddsThreshold);
        }
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#map(KEYIN, VALUEIN, org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void map(LongWritable key, Text value, Context context)
        		throws IOException, InterruptedException {
        	items  =  value.toString().split(fieldDelimRegex);
        	logOdds = 0;
        	if (items.length >= (skipFieldCount + 2)) {
	        	for (int i = skipFieldCount + 1; i < items.length; ++i) {
	        		//cumulative log odds for 2 classes based on respective state transition probability matrix
	        		frState = items[i-1];
	        		toState = items[i];
	        		logOdds += Math.log((double)model.getStateTransProbability(classLabels[0], frState, toState) / 
	        				(double)model.getStateTransProbability(classLabels[1], frState, toState));
	        	}
	        	predClass = logOdds > logOddsThreshold ? classLabels[0] : classLabels[1];
	        	stBld.delete(0, stBld.length());
	        	stBld.append(items[idFieldOrd]).append(fieldDelim);
	        	if (inValidationMode){
	        		stBld.append(items[classLabelFieldOrd]).append(fieldDelim);
	        	}
	        	stBld.append(predClass).append(fieldDelim).append(logOdds);
	        	
	        	outVal.set(stBld.toString());
    			context.write(NullWritable.get(),outVal);
        	}
        }        
	}	
	
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new MarkovModelClassifier(), args);
        System.exit(exitCode);
	}
	

}
