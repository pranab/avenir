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
 * UCB2 algorithm
 * @author pranab
 *
 */
public class UpperConfidenceBoundTwoLearner extends ReinforcementLearner {
	private Map<String, SimpleStat> rewardStats = new HashMap<String, SimpleStat>();
	private Map<String, Integer> numEpochs = new HashMap<String, Integer>();
	private int rewardScale;
	private double alpha;
	private Action currentAction;
	private int epochSize;
	private int epochTrialCount;

	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		rewardScale = ConfigUtility.getInt(config, "reward.scale",  100);
		alpha = ConfigUtility.getDouble(config, "ucb2.alpha", 0.1);
        for (Action action : actions) {
        	rewardStats.put(action.getId(), new SimpleStat());
        	numEpochs.put(action.getId(), 0);
        }
	}

	@Override
	public Action[] nextActions(int roundNum) {
		Action action = null;
		double score = 0;
		
		//check for min trial requirement
		action = selectActionBasedOnMinTrial();
		
		if (null == action) {
			if (null != currentAction && epochTrialCount < epochSize) {
				//continue with current epoch
				action = currentAction;
				++epochTrialCount;
			} else {
				if (null != currentAction) {
					numEpochs.put(currentAction.getId(), numEpochs.get(currentAction.getId()) + 1);
				}
				
				//start epoch with another action
		        for (Action thisAction : actions) {
		        	double thisReward = (rewardStats.get(thisAction.getId()).getMean());
		        	int epochCount = numEpochs.get(thisAction.getId());
		        	double tao = epochCount == 0 ? 1.0 : Math.pow((1.0 + alpha), epochCount);
		        	double a = (1 + alpha) * Math.log(Math.E * totalTrialCount / tao) / (2 * tao);
		        	double thisScore = thisReward + Math.sqrt(a);
		        	if (thisScore >  score) {
		        		score = thisScore;
		        		action = thisAction;
		        	}
		        }
		        
		        //start new epoch
		        currentAction = action;
		        int epochCount = numEpochs.get(action.getId());
		        epochSize = (int)Math.round(Math.pow((1.0 + alpha), (epochCount + 1)) - Math.pow((1.0 + alpha), epochCount));
		        epochSize = epochSize == 0 ? 1 : epochSize;
		        epochTrialCount = 0;
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
