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
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.explore.ClassPartitionGenerator.PartitionGeneratorReducer;

/**
 * Info stat for splits based various statistical criteria, to enable selection
 * of optimum split for  attributes
 * @author pranab
 *
 */
public class AttributeSplitStat {
	private int attrOrdinal;
	private Map<String, SplitStat> splitStats = new HashMap<String, SplitStat>();
    private static final Logger LOG = Logger.getLogger(AttributeSplitStat.class);
    private Set<String> classValues = new HashSet<String>();
    public static final String ALG_ENTROPY = "entropy";
    public static final String ALG_GINI_INDEX = "giniIndex";
    public static final String ALG_HELLINGER_DIST = "hellingerDistance";
    public static final String ALG_CLASS_CONF = "classConfidenceRatio";
    private String algorithm;
	
    public static void enableLog() {
    	LOG.setLevel(Level.DEBUG);
    }
    
	/**
	 * @param attrOrdinal
	 * @param algorithm
	 */
	public AttributeSplitStat(int attrOrdinal, String algorithm) {
		this.attrOrdinal = attrOrdinal;
		this.algorithm = algorithm;
	}
	
	/**
	 * @param key
	 * @param segmentIndex
	 * @param classVal
	 * @param count
	 */
	public void countClassVal(String key, int segmentIndex, String classVal, int count) {
		SplitStat splitStat = splitStats.get(key);
		if (null == splitStat) {
			if (algorithm.equals(ALG_ENTROPY) || algorithm.equals(ALG_GINI_INDEX)) {
				splitStat = new SplitInfoContent(key);
			} else if (algorithm.equals(ALG_HELLINGER_DIST)){
				splitStat = new SplitHellingerDistance(key);
			} else {
				splitStat = new SplitClassCofidenceRatio(key);
			}
			splitStats.put(key, splitStat);
		}
		splitStat.countClassVal(segmentIndex, classVal, count);
		classValues.add(classVal);
	}
	
	/**
	 * @param algorithm
	 * @return
	 */
	public Map<String, Double> processStat(String algorithm) {
		Map<String, Double> stats =new HashMap<String, Double>();
		
		for (String key : splitStats.keySet()) {
			SplitStat splitStat = splitStats.get(key);
			stats.put(key,  splitStat.processStat(algorithm, classValues));
		}
		return stats;
	}
	
	/**
	 * @param splitKey
	 * @return
	 */
	public Map<Integer, Map<String, Double>> getClassProbab(String splitKey) {
		SplitStat splitStat = splitStats.get(splitKey);
		return splitStat.getClassProbab();
	}
	
	/**
	 * @param splitKey
	 * @return
	 */
	public  double getInfoContent(String splitKey) {
		SplitStat splitStat = splitStats.get(splitKey);
		return splitStat.getInfoContent();
	}
	
	/**
	 * Stats for a split which consists of multiple segments
	 * @author pranab
	 *
	 */
	private static abstract class SplitStat {
		protected String key;
		protected Map<Integer, SplitStatSegment> segments = new HashMap<Integer, SplitStatSegment>();
		
		public SplitStat(String key) {
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
		
		public abstract double processStat(String algorithm, Set<String> classValues);
		
		/**
		 * @return
		 */
		public Map<Integer, Map<String, Double>>  getClassProbab() {
			Map<Integer, Map<String, Double>> classProbab = new HashMap<Integer, Map<String, Double>>();
			for (Integer segmentIndex : segments.keySet()) {
				SplitStatSegment statSegment = segments.get(segmentIndex);
				classProbab.put(segmentIndex, statSegment.getClassValPr());
			}		
			return classProbab;
		}
		
		/**
		 * @return
		 */
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
	 * Entropy or gini index based split stat
	 * @author pranab
	 * Entropy or gini index based
	 */
	private static class SplitInfoContent extends SplitStat {
		/**
		 * @param key
		 */
		public SplitInfoContent(String key) {
			super(key);
			LOG.debug("constructing SplitInfoContent key:" + key);
		}
		
		/* (non-Javadoc)
		 * @see org.avenir.util.AttributeSplitStat.SplitStat#processStat(java.lang.String, java.util.Set)
		 */
		public double processStat(String algorithm, Set<String> classValues) {
			double stats = 0;
			LOG.debug("processing SplitStat key:" + key);
		
			double[] statArr = new double[segments.size()];
			int[] countArr = new int[segments.size()];
			int totalCount = 0;
			int i = 0;
			for (Integer segmentIndex : segments.keySet()) {
				SplitStatSegment statSegment = segments.get(segmentIndex);
				double stat = statSegment.processStat(algorithm);
				statArr[i] = stat;
				int count = statSegment.getTotalCount();
				countArr[i] = count;
				totalCount += count;
				++i;
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
	}

	
	/**
	 * Hellinger distance base split stat
	 * @author pranab
	 * Hellinger distance based crtieria
	 *
	 */
	private static class SplitHellingerDistance extends SplitStat {
		/**
		 * @param key
		 */
		public SplitHellingerDistance(String key) {
			super(key);
			LOG.debug("constructing SplitHellingerDistance key:" + key);
		}		
		
		/* (non-Javadoc)
		 * @see org.avenir.util.AttributeSplitStat.SplitStat#processStat(java.lang.String, java.util.Set)
		 */
		public double processStat(String algorithm, Set<String> classValues) {
			double stats = 0;
			LOG.debug("processing SplitStat key:" + key);
		
			if (classValues.size() != 2) {
				throw new IllegalArgumentException(
						"Hellinger distance algorithm is only valid for binary valued class attributes");
			}
			
			//binary class values
			String[] classValueArr = new String[2];
			int ci = 0;
			for (String classVal : classValues) {
				classValueArr[ci++] = classVal;
			}
			
			//class value counts
			int[] classValCount = new int[2];
			for (int i = 0; i < 2; ++i) {
				classValCount[i] = 0;
				for (Integer segmentIndex : segments.keySet()) {
					SplitStatSegment statSegment = segments.get(segmentIndex);
					classValCount[i] += statSegment.getCountForClassVal(classValueArr[i]);
				}
			}
			
			//hellinger distance
			double sum = 0;
			for (Integer segmentIndex : segments.keySet()) {
				SplitStatSegment statSegment = segments.get(segmentIndex);
				double val0 = (double)statSegment.getCountForClassVal(classValueArr[0]) / classValCount[0];
				statSegment.setClassConfidence(classValueArr[0], val0);
				val0 = Math.sqrt(val0);
				double val1 = (double)statSegment.getCountForClassVal(classValueArr[1]) / classValCount[1];
				statSegment.setClassConfidence(classValueArr[1], val1);
				val1 = Math.sqrt(val1);
				sum += (val0 - val1) * (val0 - val1);
			}				
			stats = Math.sqrt(sum);
			
			LOG.debug("split key:" + key + " stats:" +  stats);
			return stats;
		}
		
	}	

	/**
	 * Confidence ration based split stat
	 * @author pranab
	 * Class confidence based criteria
	 */
	private static class SplitClassCofidenceRatio extends SplitStat {
		public SplitClassCofidenceRatio(String key) {
			super(key);
			LOG.debug("constructing SplitClassCofidenceRatio key:" + key);
		}

		@Override
		public double processStat(String algorithm, Set<String> classValues) {
			Map <String, Integer> classValCount = new HashMap <String, Integer>();
			double stat = 0;
			
			//class attribute total count
			for (String classVal : classValues) {
				for (Integer segmentIndex : segments.keySet()) {
					SplitStatSegment statSegment = segments.get(segmentIndex);
					int count = statSegment.getCountForClassVal(classVal);
					if ( null == classValCount.get(classVal)) {
						classValCount.put(classVal, 0);
					}
					classValCount.put(classVal, classValCount.get(classVal) + count);
				}				
			}
			
			// class confidence
			for (String classVal : classValues) {
				for (Integer segmentIndex : segments.keySet()) {
					SplitStatSegment statSegment = segments.get(segmentIndex);
					int count = statSegment.getCountForClassVal(classVal);
					int totalCount = classValCount.get(classVal);
					statSegment.setClassConfidence(classVal, (double)count / totalCount);
				}				
			}

			//class cofidence ratio
			int totalCount = 0;
			double sum = 0;
			for (Integer segmentIndex : segments.keySet()) {
				SplitStatSegment statSegment = segments.get(segmentIndex);
				double classConfRatio  = statSegment.processClassConfidenceRatio();
				int count = statSegment.getTotalCount();
				sum += classConfRatio * count;
				totalCount += count;
			}
			stat = sum / totalCount;
			return stat;
		}		
		
		
	}	
	
	/** Stats for a split segment i.e. range for numerical attribute and group of 
	 * of attributes for categorical
	 * @author pranab
	 *
	 */
	private static class SplitStatSegment {
		private int segmentIndex;
		private Map<String, Integer> classValCount = new HashMap<String, Integer>();
		private Map<String, Double> classValPr = new HashMap<String, Double>();
		private Map<String, Double> classValConfidence = new HashMap<String, Double>();
		private Map<String, Double> classValConfidenceRatio = new HashMap<String, Double>();
		private int totalCount = 0;
		
		/**
		 * @param segmentIndex
		 */
		public SplitStatSegment(int segmentIndex) {
			LOG.debug("constructing SplitStatSegment segmentIndex:" + segmentIndex);
			this.segmentIndex = segmentIndex;
		}
		
		/**
		 * @param classVal
		 * @param count
		 */
		public void countClassVal(String classVal, int count) {
			LOG.debug("counting SplitStatSegment segmentIndex:" + segmentIndex + 
					" classVal:" + classVal + " count:" + count);
			if (null == classValCount.get(classVal)) {
				classValCount.put(classVal, 0);
			}
			classValCount.put(classVal, classValCount.get(classVal) + count);
		}
		
		/**
		 * @param algorithm
		 * @return
		 */
		public double processStat(String algorithm) {
			double stat = 0.0;
			totalCount = 0;
			for (String key : classValCount.keySet()) {
				totalCount += classValCount.get(key);
			}
			LOG.debug("processing segment index:" + segmentIndex + " total count:" + totalCount);
			
			if (algorithm.equals(ALG_ENTROPY)) {
				//entropy based
				double log2 = Math.log(2);
				for (String key : classValCount.keySet()) {
					double pr = (double)classValCount.get(key) / totalCount;
					stat -= pr * Math.log(pr) / log2;
					classValPr.put(key, pr);
				}
				
			} else if (algorithm.equals(ALG_GINI_INDEX)) {
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

		/**
		 * @return
		 */
		public int getTotalCount() {
			if (0 == totalCount) {
				for (String key : classValCount.keySet()) {
					totalCount += classValCount.get(key);
				}
			}
			return totalCount;
		}

		/**
		 * @return
		 */
		public Map<String, Double> getClassValPr() {
			return classValPr;
		}
		
		/**
		 * @param classVal
		 * @return
		 */
		public int getCountForClassVal(String classVal) {
			Integer countObj = classValCount.get(classVal);
			return countObj == null? 0 : countObj;
		}
		
		/**
		 * @param classVal
		 * @param confidence
		 */
		public void setClassConfidence(String classVal, double confidence) {
			classValConfidence.put(classVal, confidence);
		}
		
		/**
		 * @return
		 */
		public double processClassConfidenceRatio() {
			double entropy = 0;
			double totalClassConf = 0;
			for (String classVal : classValConfidence.keySet()) {
				totalClassConf += classValConfidence.get(classVal);
			}
			for (String classVal : classValConfidence.keySet()) {
				double classConfRatio =  classValConfidence.get(classVal) / totalClassConf;
				classValConfidenceRatio.put(classVal, classConfRatio);
			}
			
			//entropy based on class confidence ratio
			double log2 = Math.log(2);
			for (String key : classValConfidenceRatio.keySet()) {
				double ccr = classValConfidenceRatio.get(key);
				entropy -= ccr * Math.log(ccr) / log2;
			}
			
			return entropy;
		}
	}

}
