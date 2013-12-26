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


package org.avenir.reinforce;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Optimistic sampson sampler
 * @author pranab
 *
 */
public class OptimisticSampsonSampler extends SampsonSampler {
	private Map<String, Integer> meanRewards = new HashMap<String, Integer>();
	
	/**
	 * @param actionID
	 */
	public void computeRewardMean(String actionID) {
		List<Integer> rewards = rewardDistr.get(actionID);
		if (null != rewards) {
			int sum = 0;
			int count = 0;
			for (int reward : rewards) {
				sum += reward;
				++count;
			}
			meanRewards.put(actionID, sum/count);
		}
	}
	
	public int enforce(String actionID, int reward) {
		int meanReward = meanRewards.get(actionID);
		return reward > meanReward ? reward : meanReward;
	}
	
}
