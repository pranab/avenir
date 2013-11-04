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

import org.apache.log4j.Logger;

/**
 * Info content based stat for whole data set without split
 * @author pranab
 *
 */
public class InfoContentStat {
	private Map<String, Integer> classValCount = new HashMap<String, Integer>();
	private Map<String, Double> classValPr = new HashMap<String, Double>();
	private int totalCount;
    private static final Logger LOG = Logger.getLogger(InfoContentStat.class);
	
	
	/**
	 * @param classVal
	 * @param count
	 */
	public void countClassVal(String classVal, int count) {
		LOG.debug("counting InfoContentStat " + 
				" classVal:" + classVal + " count:" + count);
		if (null == classValCount.get(classVal)) {
			classValCount.put(classVal, 0);
		}
		classValCount.put(classVal, classValCount.get(classVal) + count);
	}
	
	/**
	 * Calculate info stat
	 * @param isAlgoEntropy
	 * @return
	 */
	public double processStat(boolean isAlgoEntropy) {
		double stat = 0.0;
		totalCount = 0;
		for (String key : classValCount.keySet()) {
			totalCount += classValCount.get(key);
		}
		LOG.debug("processing total count:" + totalCount);
		
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
