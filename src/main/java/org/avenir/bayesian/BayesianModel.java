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
import org.chombo.util.BinCount;
import org.chombo.util.FeatureCount;

/**
 * Bayesian model related probability distributions
 * @author pranab
 *
 */
public class BayesianModel {
	private List<FeaturePosterior> featurePosteriors = new ArrayList<FeaturePosterior>();
	private List<FeatureCount> featurePriors = new ArrayList<FeatureCount>();
	private int count;
	
	/**
	 * @param classValue
	 * @return
	 */
	public double getClassPriorProb(String classValue) {
		FeaturePosterior feaPost = getFeaturePosterior(classValue);
		return feaPost.getProb();
	}
	
	/**
	 * @param featureValues
	 * @return
	 */
	public double getFeaturePriorProb(List<Pair<Integer, Object>> featureValues) {
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
	
	/**
	 * @param classVal
	 * @param featureValues
	 * @return
	 */
	public double getFeaturePostProb(String classVal, List<Pair<Integer, Object>> featureValues) {
		FeaturePosterior feaPost = getFeaturePosterior(classVal);
		double prob = feaPost.getFeaturePostProb(featureValues);
		return prob;
	}
	
	/**
	 * @param classValue
	 * @param count
	 */
	public void addClassPrior(String classValue, int count) {
		FeaturePosterior feaPost = getFeaturePosterior(classValue);
		feaPost.addCount(count);
	}
	
	/**
	 * @param ordinal
	 * @param bin
	 * @param count
	 */
	public void addFeaturePrior(int ordinal, String bin,  int count) {
		FeatureCount feaCount = getFeatureCount( ordinal);
		BinCount binCount = new BinCount(bin, count);
		feaCount.addBinCount(binCount);
	}
	
	/**
	 * @param ordinal
	 * @param mean
	 * @param stdDev
	 */
	public void setFeaturePriorParaemeters(int ordinal, long mean, long stdDev) {
		FeatureCount feaCount = getFeatureCount( ordinal);
		feaCount.setDistrParameters(mean, stdDev);
	}	
	
	/**
	 * @param classValue
	 * @param ordinal
	 * @param bin
	 * @param count
	 */
	public void addFeaturePosterior(String classValue, int ordinal, String bin,  int count) {
		FeaturePosterior feaPost = getFeaturePosterior(classValue);
		FeatureCount  feaCount =  feaPost.getFeatureCount( ordinal);
		BinCount binCount = new BinCount(bin, count);
		feaCount.addBinCount(binCount);
	}
	
	/**
	 * @param classValue
	 * @param ordinal
	 * @param mean
	 * @param stdDev
	 */
	public void setFeaturePosteriorParaemeters(String classValue, int ordinal, long mean, long stdDev) {
		FeaturePosterior feaPost = getFeaturePosterior(classValue);
		FeatureCount  feaCount =  feaPost.getFeatureCount(ordinal);
		feaCount.setDistrParameters(mean, stdDev);
	}	
	
	/**
	 * @param ordinal
	 * @return
	 */
	private FeatureCount getFeatureCount(int ordinal) {
		FeatureCount feaCount  = null;
		for (FeatureCount thisFeaCount :   featurePriors) {
			if (thisFeaCount.getOrdinal() == ordinal) {
				feaCount = thisFeaCount;
				break;
			}
		}
		if (null ==  feaCount) {
			feaCount = new FeatureCount(ordinal, "");
			featurePriors.add(feaCount);
		}
		return feaCount;
	}
	
	/**
	 * @param classValue
	 * @return
	 */
	private FeaturePosterior getFeaturePosterior(String classValue) {
		FeaturePosterior feaPost = null;
		for (FeaturePosterior thisFeaPost  :  featurePosteriors) {
			if (thisFeaPost.getClassValue().equals(classValue)) {
				feaPost = thisFeaPost;
				break;
			}
		}
		
		if (null == feaPost) {
			feaPost = new FeaturePosterior();
			feaPost.setClassValue(classValue);
			featurePosteriors.add(feaPost);
		}
		
		return feaPost;
	}
	
	/**
	 * @return
	 */
	public List<FeaturePosterior> getFeaturePosteriors() {
		return featurePosteriors;
	}
	
	/**
	 * @param featurePosteriors
	 */
	public void setFeaturePosteriors(List<FeaturePosterior> featurePosteriors) {
		this.featurePosteriors = featurePosteriors;
	}
	
	/**
	 * @return
	 */
	public List<FeatureCount> getFeaturePriors() {
		return featurePriors;
	}
	
	/**
	 * @param featurePriors
	 */
	public void setFeaturePriors(List<FeatureCount> featurePriors) {
		this.featurePriors = featurePriors;
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
	 * 
	 */
	public void finishUp() {
		//total count by adding all class prior counts
		count = 0;
		for (FeaturePosterior thisFeaPost  :  featurePosteriors) {
			count += thisFeaPost.getCount();
		}	
		
		//class prior and feature posterior
		for (FeaturePosterior thisFeaPost  :  featurePosteriors) {
			thisFeaPost.normalize(count);
		}		
		
		//feature prior
		for (FeatureCount thisFeaCount :   featurePriors) {
			thisFeaCount.normalize(count);
		}
	}
}
