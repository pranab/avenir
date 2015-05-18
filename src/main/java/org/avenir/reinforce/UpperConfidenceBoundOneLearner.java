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

import org.chombo.util.ConfigUtility;
import org.chombo.util.SimpleStat;

/**
 * UCB1 
 * @author pranab
 *
 */
public class UpperConfidenceBoundOneLearner extends ReinforcementLearner {
	private Map<String, SimpleStat> rewardStats = new HashMap<String, SimpleStat>();
	private int rewardScale;

	@Override
	public void initialize(Map<String, Object> config) {
		rewardScale = ConfigUtility.getInt(config, "reward.scale",  100);
        for (Action action : actions) {
        	rewardStats.put(action.getId(), new SimpleStat());
        }
	}

	@Override
	public Action[] nextActions(int roundNum) {
		Action action = null;
		double score = 0;
        for (Action thisAction : actions) {
        	double thisReward = (rewardStats.get(thisAction.getId()).getMean());
        	double thisScore = thisReward + Math.sqrt(2.0 * Math.log(totalTrialCount) / thisAction.getTrialCount());
        	if (thisScore >  score) {
        		score = thisScore;
        		action = thisAction;
        	}
        }
		
		++totalTrialCount;
		action.select();
		selActions[0] = action;
		return selActions;
	}

	@Override
	public void setReward(String actionId, int reward) {
		double dReward = (double)reward / rewardScale;
		rewardStats.get(actionId).add(dReward);
		findAction(actionId).reward(reward);
	}

}
