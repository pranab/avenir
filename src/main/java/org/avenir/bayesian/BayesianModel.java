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

import org.chombo.util.BinCount;
import org.chombo.util.FeatureCount;

public class BayesianModel {
	private List<FeaturePosterior> featurePosteriors = new ArrayList<FeaturePosterior>();
	private List<FeatureCount> featurePriors = new ArrayList<FeatureCount>();
	private int count;
	
	public void addClassPrior(String classValue, int count) {
		FeaturePosterior feaPost = getFeaturePosterior(classValue);
		feaPost.setCount(count);
	}
	
	public void addFeaturePrior(int ordinal, String bin,  int count) {
		FeatureCount feaCount = getFeatureCount( ordinal);
		BinCount binCount = new BinCount(bin, count);
		feaCount.addBinCount(binCount);
	}
	
	public void addFeaturePosterior(String classValue, int ordinal, String bin,  int count) {
		FeaturePosterior feaPost = getFeaturePosterior(classValue);
		FeatureCount  feaCount =  feaPost.getFeatureCount( ordinal);
		BinCount binCount = new BinCount(bin, count);
		feaCount.addBinCount(binCount);
	}
	
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
	
	public List<FeaturePosterior> getFeaturePosteriors() {
		return featurePosteriors;
	}
	public void setFeaturePosteriors(List<FeaturePosterior> featurePosteriors) {
		this.featurePosteriors = featurePosteriors;
	}
	public List<FeatureCount> getFeaturePriors() {
		return featurePriors;
	}
	public void setFeaturePriors(List<FeatureCount> featurePriors) {
		this.featurePriors = featurePriors;
	}
	public int getCount() {
		return count;
	}
	public void setCount(int count) {
		this.count = count;
	}
}
