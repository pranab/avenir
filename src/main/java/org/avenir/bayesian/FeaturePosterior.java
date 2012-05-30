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

import org.chombo.util.FeatureCount;

public class FeaturePosterior {
	private String classValue;
	private List<FeatureCount> featureCounts = new ArrayList<FeatureCount>();
	private int count;
	
	public String getClassValue() {
		return classValue;
	}
	public void setClassValue(String classValue) {
		this.classValue = classValue;
	}
	public List<FeatureCount> getFeatureCounts() {
		return featureCounts;
	}
	public void setFeatureCounts(List<FeatureCount> featureCounts) {
		this.featureCounts = featureCounts;
	}
	public int getCount() {
		return count;
	}
	public void setCount(int count) {
		this.count = count;
	}
	public  FeatureCount getFeatureCount(int ordinal) {
		FeatureCount feaCount  = null;
		for (FeatureCount thisFeaCount :   featureCounts) {
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

}
