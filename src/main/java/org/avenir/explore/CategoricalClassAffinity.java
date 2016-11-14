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
import java.util.Collections;
import java.util.Comparator;
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
import org.chombo.util.BasicUtils;
import org.chombo.util.Pair;
import org.chombo.util.Tuple;
import org.chombo.util.Utility;

/**
 * Calculates affinity of categorical attribute values to class values
 * @author pranab
 *
 */
public class CategoricalClassAffinity extends Configured implements Tool {
	private static String STRAT_ODDS_RATIO = "oddsRatio";
	private static String STRAT_DISTR_DIFF = "distrDiff";
	private static String STRAT_MIN_RISK = "minRisk";
	private static String STRAT_KL_DIFF = "klDiff";
	

	@Override
	public int run(String[] args) throws Exception {
        Job job = new Job(getConf());
        String jobName = "Categorical attribute affinity to class values  MR";
        job.setJobName(jobName);
        
        job.setJarByClass(CategoricalClassAffinity.class);

        FileInputFormat.addInputPaths(job, args[0]);
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        Utility.setConfiguration(job.getConfiguration());
        
        job.setMapperClass(CategoricalClassAffinity.AffinityMapper.class);
        job.setReducerClass(CategoricalClassAffinity.AffinityReducer.class);

        job.setMapOutputKeyClass(Tuple.class);
        job.setMapOutputValueClass(Tuple.class);

        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        Utility.setTuplePairSecondarySorting(job);
        
        int numReducer = job.getConfiguration().getInt("cca.num.reducer", -1);
        numReducer = -1 == numReducer ? job.getConfiguration().getInt("num.reducer", 1) : numReducer;
        job.setNumReduceTasks(numReducer);
        
        int status =  job.waitForCompletion(true) ? 0 : 1;
        return status;
	}


	/**
	 * @author pranab
	 *
	 */
	public static class AffinityMapper extends Mapper<LongWritable, Text, Tuple, Tuple> {
		private Tuple outKey = new Tuple();
		private Tuple outVal = new Tuple();
        private String fieldDelimRegex;
        private String[] items;
        private String posClassAttrValue;
        private String classAttrValue;
        private int featureOrd;
        private String featureVal;
        private double featureDistr;
        
        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimRegex = config.get("field.delim.regex", ",");
        	posClassAttrValue = Utility.assertStringConfigParam(config, "cca.pos.class.attr.value", 
        			"missing positive class attribute value");
        }	
	    
        @Override
	    protected void map(LongWritable key, Text value, Context context)
	    	throws IOException, InterruptedException {
        	items  = value.toString().split(fieldDelimRegex, -1);
        	
        	featureOrd = BasicUtils.extractIntFromStringArray(items, 0);
        	classAttrValue = items[1];
        	featureVal = items[2];
        	featureDistr = BasicUtils.extractDoubleFromStringArray(items, 3);
        	outKey.initialize();
        	outVal.initialize();
        	
        	if (classAttrValue.equals(posClassAttrValue)) {
        		outKey.add(featureOrd, 0);
        	} else {
        		outKey.add(featureOrd, 1);
        	}
        	outVal.add(classAttrValue, featureVal, featureDistr);
        	context.write(outKey, outVal);
        }	
	}
	
    /**
     * @author pranab
     *
     */
    public static class AffinityScore extends Pair<String, Double> {
    	public AffinityScore(String feature, Double score) {
    		super(feature, score);
    	}
    }
    
    /**
     * @author pranab
     *
     */
    public static class AffinityScoreComparator  implements Comparator<AffinityScore>  {
		@Override
		public int compare(AffinityScore thisAffinity, AffinityScore thatAffinity) {
			return thisAffinity.getRight() < thatAffinity.getRight() ? 1 :
				(thisAffinity.getRight() > thatAffinity.getRight() ? -1 : 0);
		}
    }
	
    /**
     * @author pranab
     *
     */
    public static class AffinityReducer extends Reducer<Tuple, Tuple, NullWritable, Text> {
		private Text outVal = new Text();
		private String fieldDelimOut;
		private StringBuilder stBld = new  StringBuilder();
        private String[] affinityStrategies;
        private List<Map<String, Double>> featureClassCondDistr = new ArrayList<Map<String, Double>>();
        private List<AffinityScore> affinityScores = new ArrayList<AffinityScore>();
        private AffinityScoreComparator scoreComparator = new AffinityScoreComparator();

        /* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void setup(Context context) throws IOException, InterruptedException {
        	Configuration config = context.getConfiguration();
        	fieldDelimOut = config.get("field.delim", ",");
        	affinityStrategies = Utility.assertStringArrayConfigParam(config, "cca.affinity.strategy",
        			Utility.DEF_FIELD_DELIM, "missing affinity strategy list");
           	for (int i = 0; i < 2; ++i){
           		featureClassCondDistr.add(new HashMap<String, Double>());
           	}
        }
    
		/* (non-Javadoc)
         * @see org.apache.hadoop.mapreduce.Reducer#reduce(KEYIN, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
         */
        protected void reduce(Tuple key, Iterable<Tuple> values, Context context)
        	throws IOException, InterruptedException {
        	
        	String classVal = null;
        	Map<String, Double> distr = null;
        	for (Tuple value : values){
    			String thisClassVal = value.getString(0);
        		if (null == classVal) {
        			classVal = thisClassVal;
        			distr = featureClassCondDistr.get(0);
        			distr.clear();
        		} else {
        			if (!thisClassVal.equals(classVal)) {
        				//switching to other class value
        				classVal = thisClassVal;
        				distr  = featureClassCondDistr.get(1);
            			distr.clear();
        			}
        		}
				distr.put(value.getString(1), value.getDouble(2));
        	}
        	
        	//all algorithms
        	for (String  affinityStrategy : affinityStrategies) {
        		outVal.set("algorithm: " + affinityStrategy);
	        	context.write(NullWritable.get(), outVal);
        		
	        	//affinity score 
	        	affinityScores.clear();
	        	findAffinityScore(affinityStrategy);
	        	
	        	//sort by score
	        	Collections.sort(affinityScores, scoreComparator);
	        	
	        	//emit
	        	stBld.delete(0, stBld.length());
	        	for (AffinityScore score : affinityScores) {
	        		stBld.append(key.get(0)).append(fieldDelimOut).append(score.getLeft()).
	        			append(fieldDelimOut).append(score.getRight()).append("\n");
	        	}
	        	outVal.set(stBld.substring(0, stBld.length() -1));
	        	context.write(NullWritable.get(), outVal);
        	}
        }
        
        /**
         * 
         */
        private void  findAffinityScore(String  affinityStrategy) {
        	Map<String, Double> posDistr = featureClassCondDistr.get(0);
        	Map<String, Double> negDistr = featureClassCondDistr.get(1);
        	for (String featureVal : posDistr.keySet()) {
        		double pDistr = posDistr.get(featureVal);
        		double nDistr = negDistr.get(featureVal);
        		
        		if (affinityStrategy.equals(STRAT_ODDS_RATIO)) {
        			double oddsRatio = (pDistr / (1 - pDistr)) / (nDistr / (1 - nDistr));
        			affinityScores.add(new AffinityScore(featureVal, oddsRatio));
        		} else if (affinityStrategy.equals(STRAT_DISTR_DIFF)) {
        			double diff = pDistr - nDistr;
        			affinityScores.add(new AffinityScore(featureVal, diff));
        		} else if (affinityStrategy.equals(STRAT_MIN_RISK)) {
        			double minRisk = pDistr * (1 - nDistr);
        			affinityScores.add(new AffinityScore(featureVal, minRisk));
        		} else if (affinityStrategy.equals(STRAT_KL_DIFF)) {
        			double kl = pDistr * Math.log(pDistr / nDistr);
        			affinityScores.add(new AffinityScore(featureVal, kl));
        		}
        	}
        }
    }	
    
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new CategoricalClassAffinity(), args);
        System.exit(exitCode);
	}
    
}
