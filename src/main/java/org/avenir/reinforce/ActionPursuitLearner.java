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

import org.chombo.util.CategoricalSampler;
import org.chombo.util.ConfigUtility;
import org.chombo.util.SimpleStat;


/**
 * Action pursuit larner
 * @author pranab
 *
 */
public class ActionPursuitLearner extends ReinforcementLearner {
	private double learningRate;
	private CategoricalSampler sampler = new CategoricalSampler();
	private String rewardedAction;
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		learningRate  = ConfigUtility.getDouble(config, "pursuit.learning.rate", 0.05);
        
		double intialProb = 1.0 / actions.size();
        for (Action action : actions) {
        	sampler.add(action.getId(), intialProb);
        }
 	}
	
	
	@Override
	public Action[] nextActions(int roundNum) {
		for (int i = 0; i < batchSize; ++i) {
			selActions[i] = nextAction(roundNum + i);
		}
		return selActions;
	}

	/**
	 * @param roundNum
	 * @return
	 */
	public Action nextAction(int roundNum) {
		Action action = null;
		double distr = 0;
		if (null != rewardedAction) {
	        for (Action thisAction : actions) {
        		distr = sampler.get(thisAction.getId());
	        	if (thisAction.equals(rewardedAction)) {
	        		distr += learningRate * (1.0 - distr);
	        	} else {
	        		distr -= learningRate * distr;
	        	}
        		sampler.set(thisAction.getId(), distr);
	        }	
	        rewardedAction = null;
		}
        action = findAction(sampler.sample());

		++totalTrialCount;
		action.select();
		return action;
	}
	
	@Override
	public void setReward(String actionId, int reward) {
		rewardedAction = actionId;
		findAction(actionId).reward(reward);
	}

}
