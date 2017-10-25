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
import java.util.Map;

import org.chombo.stats.MeanStat;
import org.chombo.stats.SimpleStat;
import org.chombo.util.ConfigUtility;

/**
 * UCB1 
 * @author pranab
 *
 */
public class UpperConfidenceBoundOneLearner extends MultiArmBanditLearner {
	private int rewardScale;

	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		populateMeanRewardStats();
	}

	/**
	 * @return
	 */
	@Override
	public Action nextAction() {
		Action action = null;
		double score = 0;
		++totalTrialCount;
		
		//check for min trial requirement
		action = selectActionBasedOnMinTrial();
		
		if (null == action) {
	        for (Action thisAction : actions) {
	        	double thisReward = (meanRewardStats.get(thisAction.getId()).getMean());
	        	double thisScore = thisReward + Math.sqrt(2.0 * Math.log(totalTrialCount) / thisAction.getTrialCount());
	        	if (thisScore >  score) {
	        		score = thisScore;
	        		action = thisAction;
	        	}
	        }
		}
		action.select();
		return action;
	}

	@Override
	public void setReward(String actionId, double reward) {
		double scaledReward = reward / rewardScale;
		meanRewardStats.get(actionId).add(scaledReward);
		findAction(actionId).reward(scaledReward);
	}

	@Override
	public void buildModel(String model) {
		buildMeanRewardStatModel(model);
	}

	@Override
	public String[] getModel() {
		return getMeanRewardStatModel();
	}
}
