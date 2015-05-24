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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chombo.util.ConfigUtility;
import org.chombo.util.Utility;

/**
 * Sampson sampler probabilistic matching reinforcement learning
 * @author pranab
 *
 */
public class SampsonSampler extends ReinforcementLearner {
	protected  Map<String, List<Integer>> rewardDistr = new HashMap<String, List<Integer>>();
	private int minSampleSize;
	private int maxReward;
	
	/**
	 * @param actionID
	 * @param reward
	 */
	@Override
	public void setReward(String actionID, int reward) {
		List<Integer> rewards = rewardDistr.get(actionID);
		if (null == rewards) {
			rewards = new ArrayList<Integer>();
			rewardDistr.put(actionID, rewards);
		}
		rewards.add(reward);
		findAction(actionID).reward(reward);
	}
	
	@Override
	public Action[] nextActions() {
		for (int i = 0; i < batchSize; ++i) {
			selActions[i] = nextAction();
		}
		return selActions;
	}

	/**
	 * Select action
	 * @return
	 */
	public Action  nextAction() {
		String slectedActionID = null;
		int maxRewardCurrent = 0;
		int reward = 0;
		++totalTrialCount;
		
		for (String actionID : rewardDistr.keySet()) {
			List<Integer> rewards = rewardDistr.get(actionID);
			if (rewards.size() > minSampleSize) {
				reward = Utility.selectRandom(rewards);
				reward = enforce(actionID, reward);
			} else {
				reward = (int) (Math.random() * maxReward);
			}
			
			if (reward > maxRewardCurrent) {
				slectedActionID= actionID;
				maxRewardCurrent = reward;
			}
		}
		
		Action selAction = findAction(slectedActionID);
		selAction.select();
		return selAction;
	}
	
	/**
	 * @param actionID
	 * @param reward
	 * @return
	 */
	public int enforce(String actionID, int reward) {
		return reward;
	}

	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		minSampleSize = ConfigUtility.getInt(config, "min.sample.size");
		maxReward = ConfigUtility.getInt(config, "max.reward");
	}

}
