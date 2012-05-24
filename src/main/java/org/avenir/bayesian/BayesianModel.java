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

import java.util.List;

import org.chombo.util.FeatureCount;

public class BayesianModel {
	private List<FeaturePosterior> featurePosteriors;
	private List<FeatureCount> featurePriors;
	private int count;
	
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
