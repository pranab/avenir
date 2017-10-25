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
 * UCB2 algorithm
 * @author pranab
 *
 */
public class UpperConfidenceBoundTwoLearner extends MultiArmBanditLearner {
	private Map<String, Integer> numEpochs = new HashMap<String, Integer>();
	private int rewardScale;
	private double alpha;
	private Action currentAction;
	private int epochSize;
	private int epochTrialCount;

	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		alpha = ConfigUtility.getDouble(config, "alpha", 0.1);
		populateMeanRewardStats();
        for (Action action : actions) {
        	numEpochs.put(action.getId(), 0);
        }
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
			if (null != currentAction && epochTrialCount < epochSize) {
				//continue with current epoch
				action = currentAction;
				++epochTrialCount;
			} else {
				//epoch ended
				if (null != currentAction) {
					numEpochs.put(currentAction.getId(), numEpochs.get(currentAction.getId()) + 1);
				}
				
				//start epoch with another action
		        for (Action thisAction : actions) {
		        	double thisReward = (meanRewardStats.get(thisAction.getId()).getMean());
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
