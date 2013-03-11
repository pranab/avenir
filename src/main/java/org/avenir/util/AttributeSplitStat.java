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

package org.avenir.util;

import java.util.HashMap;
import java.util.Map;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.explore.ClassPartitionGenerator.PartitionGeneratorReducer;

/**
 * Info stat for splits
 * @author pranab
 *
 */
public class AttributeSplitStat {
	private int attrOrdinal;
	private Map<String, SplitStat> splitStats = new HashMap<String, SplitStat>();
    private static final Logger LOG = Logger.getLogger(AttributeSplitStat.class);
	
    public static void enableLog() {
    	LOG.setLevel(Level.DEBUG);
    }
    
	public AttributeSplitStat(int attrOrdinal) {
		this.attrOrdinal = attrOrdinal;
	}
	
	public void countClassVal(String key, int segmentIndex, String classVal, int count) {
		SplitStat splitStat = splitStats.get(key);
		if (null == splitStat) {
			splitStat = new SplitStat(key);
			splitStats.put(key, splitStat);
		}
		splitStat.countClassVal(segmentIndex, classVal, count);
	}
	
	public Map<String, Double> processStat(boolean isAlgoEntropy) {
		Map<String, Double> stats =new HashMap<String, Double>();
		
		for (String key : splitStats.keySet()) {
			SplitStat splitStat = splitStats.get(key);
			stats.put(key,  splitStat.processStat(isAlgoEntropy));
		}
		return stats;
	}
	
	public Map<Integer, Map<String, Double>> getClassProbab(String splitKey) {
		SplitStat splitStat = splitStats.get(splitKey);
		return splitStat.getClassProbab();
	}
	
	public  double getInfoContent(String splitKey) {
		SplitStat splitStat = splitStats.get(splitKey);
		return splitStat.getInfoContent();
	}
	
	/**
	 * @author pranab
	 *
	 */
	private static class SplitStat {
		private String key;
		private Map<Integer, SplitStatSegment> segments = new HashMap<Integer, SplitStatSegment>();
		
		public SplitStat(String key) {
			LOG.debug("constructing SplitStat key:" + key);
			this.key = key;
		}
		
		public void countClassVal(int segmentIndex, String classVal, int count) {
			LOG.debug("counting  SplitStat key:" + key);
			SplitStatSegment statSegment = segments.get(segmentIndex);
			if (null == statSegment) {
				statSegment = new SplitStatSegment(segmentIndex);
				segments.put(segmentIndex, statSegment);
			}
			statSegment.countClassVal(classVal, count);
		}
		
		public double processStat(boolean isAlgoEntropy) {
			double stats = 0;
			LOG.debug("processing SplitStat key:" + key);
			
			double[] statArr = new double[segments.size()];
			int[] countArr = new int[segments.size()];
			int totalCount = 0;
			int i = 0;
			for (Integer segmentIndex : segments.keySet()) {
				SplitStatSegment statSegment = segments.get(segmentIndex);
				double stat = statSegment.processStat(isAlgoEntropy);
				statArr[i] = stat;
				int count = statSegment.getTotalCount();
				countArr[i] = count;
				totalCount += count;
			}	
			
			//weighted average
			double statSum = 0;
			for (int j = 0; j < statArr.length; ++j) {
				statSum += statArr[j] * countArr[j];
			}
			stats = statSum / totalCount;
			LOG.debug("split key:" + key + " stats:" +  stats);
			return stats;
		}
		
		public Map<Integer, Map<String, Double>>  getClassProbab() {
			Map<Integer, Map<String, Double>> classProbab = new HashMap<Integer, Map<String, Double>>();
			for (Integer segmentIndex : segments.keySet()) {
				SplitStatSegment statSegment = segments.get(segmentIndex);
				classProbab.put(segmentIndex, statSegment.getClassValPr());
			}		
			return classProbab;
		}
		
		public double getInfoContent() {
			int totalCount = 0;
			for (Integer segmentIndex : segments.keySet()) {
				SplitStatSegment statSegment = segments.get(segmentIndex);
				totalCount += statSegment.getTotalCount();
			}	
			
			double pr = 0;
			double stat = 0;
			double log2 = Math.log(2);
			for (Integer segmentIndex : segments.keySet()) {
				SplitStatSegment statSegment = segments.get(segmentIndex);
				pr = (double)statSegment.getTotalCount() / totalCount;
				stat -= pr * Math.log(pr) / log2;
			}			
			
			return stat;
		}
	}
	
	/**
	 * @author pranab
	 *
	 */
	private static class SplitStatSegment {
		private int segmentIndex;
		private Map<String, Integer> classValCount = new HashMap<String, Integer>();
		private Map<String, Double> classValPr = new HashMap<String, Double>();
		private int totalCount;
		
		public SplitStatSegment(int segmentIndex) {
			LOG.debug("constructing SplitStatSegment segmentIndex:" + segmentIndex);
			this.segmentIndex = segmentIndex;
		}
		
		public void countClassVal(String classVal, int count) {
			LOG.debug("counting SplitStatSegment segmentIndex:" + segmentIndex + 
					" classVal:" + classVal + " count:" + count);
			if (null == classValCount.get(classVal)) {
				classValCount.put(classVal, 0);
			}
			classValCount.put(classVal, classValCount.get(classVal) + count);
		}
		
		public double processStat(boolean isAlgoEntropy) {
			double stat = 0.0;
			totalCount = 0;
			for (String key : classValCount.keySet()) {
				totalCount += classValCount.get(key);
			}
			LOG.debug("processing segment index:" + segmentIndex + " total count:" + totalCount);
			
			if (isAlgoEntropy) {
				//entropy based
				double log2 = Math.log(2);
				for (String key : classValCount.keySet()) {
					double pr = (double)classValCount.get(key) / totalCount;
					stat -= pr * Math.log(pr) / log2;
					classValPr.put(key, pr);
				}
				
			} else {
				//gini index based
				double prSquare = 0;
				for (String key : classValCount.keySet()) {
					int count = classValCount.get(key);
					double pr = (double)count / totalCount;
					LOG.debug("class val:" + key + " count:" + count);
					prSquare += pr * pr;
					classValPr.put(key, pr);
				}
				stat = 1.0 - prSquare;
			}
			return stat;
		}

		public int getTotalCount() {
			return totalCount;
		}

		public Map<String, Double> getClassValPr() {
			return classValPr;
		}
	}

}
