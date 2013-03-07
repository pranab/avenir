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

/**
 * Info stat for splits
 * @author pranab
 *
 */
public class AttributeSplitStat {
	private int attrOrdinal;
	private Map<String, SplitStat> splitStats = new HashMap<String, SplitStat>();
	
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
	/**
	 * @author pranab
	 *
	 */
	private static class SplitStat {
		private String key;
		private Map<Integer, SplitStatSegment> segments = new HashMap<Integer, SplitStatSegment>();
		
		public SplitStat(String key) {
			this.key = key;
		}
		
		public void countClassVal(int segmentIndex, String classVal, int count) {
			SplitStatSegment statSegment = segments.get(segmentIndex);
			if (null == statSegment) {
				statSegment = new SplitStatSegment(segmentIndex);
			}
			statSegment.countClassVal(classVal, count);
		}
		
		public double processStat(boolean isAlgoEntropy) {
			double stats = 0;
			
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
			
			return stats;
		}
	}
	
	/**
	 * @author pranab
	 *
	 */
	private static class SplitStatSegment {
		private int segmentIndex;
		private Map<String, Integer> classValCount = new HashMap<String, Integer>();
		private int totalCount;
		
		public SplitStatSegment(int segmentIndex) {
			this.segmentIndex = segmentIndex;
		}
		
		public void countClassVal(String classVal, int count) {
			if (null == classValCount.get(classVal)) {
				classValCount.put(classVal, 0);
			}
			classValCount.put(classVal, classValCount.get(classVal) + count);
		}
		
		public double processStat(boolean isAlgoEntropy) {
			double stat = 0.0;
			int totalCount = 0;
			for (String key : classValCount.keySet()) {
				totalCount += classValCount.get(key);
			}
			
			if (isAlgoEntropy) {
				//entropy based
				double log2 = Math.log(2);
				for (String key : classValCount.keySet()) {
					double pr = (double)classValCount.get(key) / totalCount;
					stat -= pr * Math.log(pr) / log2;
				}
				
			} else {
				//gini index based
				double prSquare = 0;
				for (String key : classValCount.keySet()) {
					double pr = (double)classValCount.get(key) / totalCount;
					prSquare += pr * pr;
				}
				stat = 1.0 - prSquare;
			}
			return stat;
		}

		public int getTotalCount() {
			return totalCount;
		}
	}

}
