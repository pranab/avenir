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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.chombo.util.Pair;

/**
 * Processes mutual info score
 * @author pranab
 *
 */
public class MutualInformationScore {
	private List<FeatureClassMutualInfo>  featureClassMutualInfoList = new ArrayList<FeatureClassMutualInfo>();
	
	/**
	 * @author pranab
	 *
	 */
	public  static class FeatureClassMutualInfo extends  Pair<Double, Integer>  implements  Comparable<FeatureClassMutualInfo> {
		public FeatureClassMutualInfo(double mutualInfo, int featureOrdinal) {
			super(mutualInfo,  featureOrdinal);
		}
		
		@Override
		public int compareTo(FeatureClassMutualInfo that) {
			return that.getLeft().compareTo(this.getLeft());
		}
	}
	
	/**
	 * @param featureOrdinal
	 * @param mutualInfo
	 */
	public void addFeatureClassMutualInfo(int featureOrdinal, double mutualInfo) {
		FeatureClassMutualInfo featureClassMutualInfo = new FeatureClassMutualInfo(mutualInfo, featureOrdinal);
		featureClassMutualInfoList.add(featureClassMutualInfo);
	}
	
	/**
	 * 
	 */
	public void sortFeatureClassMutualInfo() {
		Collections.sort(featureClassMutualInfoList);
	}

	public List<FeatureClassMutualInfo> getFeatureClassMutualInfoList() {
		return featureClassMutualInfoList;
	}
	
}
