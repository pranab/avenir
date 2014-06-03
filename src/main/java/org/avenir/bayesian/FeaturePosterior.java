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

package org.avenir.bayesian;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.chombo.util.FeatureCount;

/**
 * Feature posterior probability
 * @author pranab
 *
 */
public class FeaturePosterior {
	private String classValue;
	private List<FeatureCount> featureCounts = new ArrayList<FeatureCount>();
	private int count;
	private double prob;
	
	/**
	 * @return
	 */
	public String getClassValue() {
		return classValue;
	}
	
	/**
	 * @param classValue
	 */
	public void setClassValue(String classValue) {
		this.classValue = classValue;
	}
	
	/**
	 * @return
	 */
	public List<FeatureCount> getFeatureCounts() {
		return featureCounts;
	}
	
	/**
	 * @param featureCounts
	 */
	public void setFeatureCounts(List<FeatureCount> featureCounts) {
		this.featureCounts = featureCounts;
	}
	
	/**
	 * @return
	 */
	public int getCount() {
		return count;
	}
	
	/**
	 * @param count
	 */
	public void setCount(int count) {
		this.count = count;
	}
	
	/**
	 * @param count
	 */
	public void addCount(int count) {
		this.count += count;
	}
	
	/**
	 * @param ordinal
	 * @return
	 */
	public FeatureCount getFeatureCount(int ordinal) {
		FeatureCount feaCount  = null;
		for (FeatureCount thisFeaCount :   featureCounts){
			if (thisFeaCount.getOrdinal() == ordinal) {
				feaCount = thisFeaCount;
				break;
			}
		}
		if (null ==  feaCount) {
			feaCount = new FeatureCount(ordinal, "");
			featureCounts.add(feaCount);
		}
		return feaCount;
	}

	/**
	 * @param total
	 */
	public void normalize(int total) {
		//feature posterior
		for (FeatureCount feaCount : featureCounts) {
			feaCount.normalize(count);
		}
		
		//class prior
		prob = ((double)count ) / total;
	}
	
	/**
	 * @return
	 */
	public double getProb() {
		return prob;
	}
	
	/**
	 * @param featureValues
	 * @return
	 */
	public double getFeaturePostProb( List<Pair<Integer, Object>> featureValues) {
		double prob = 1.0;
		for (Pair<Integer, Object> feature : featureValues) {
			FeatureCount feaCount = getFeatureCount( feature.getLeft());
			if (feature.getRight() instanceof String) {
				//categorical or binned numerical
				prob *= feaCount.getProb((String)feature.getRight());
			} else {
				//continuous numerical
				prob *= feaCount.getProb((Integer)feature.getRight());
			}
		}
		return prob;
	}	
}
