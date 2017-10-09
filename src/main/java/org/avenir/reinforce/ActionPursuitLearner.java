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

import java.util.Map;

import org.chombo.stats.CategoricalSampler;
import org.chombo.stats.SimpleStat;
import org.chombo.util.ConfigUtility;


/**
 * Action pursuit larner
 * @author pranab
 *
 */
public class ActionPursuitLearner extends MultiArmBanditLearner {
	private double learningRate;
	private CategoricalSampler sampler = new CategoricalSampler();
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		learningRate  = ConfigUtility.getDouble(config, "pursuit.learning.rate", 0.05);
        
		double intialProb = 1.0 / actions.size();
        for (Action action : actions) {
        	sampler.add(action.getId(), intialProb);
        	rewardStats.put(action.getId(), new SimpleStat());
        }
 	}
	
	
	/**
	 * @return
	 */
	@Override
	public Action nextAction() {
		Action action = null;
		double distr = 0;
		++totalTrialCount;

		if (rewarded) {
			Action bestAction = findBestAction();
	        for (Action thisAction : actions) {
        		distr = sampler.get(thisAction.getId());
	        	if (thisAction == bestAction) {
	        		distr += learningRate * (1.0 - distr);
	        	} else {
	        		distr -= learningRate * distr;
	        	}
        		sampler.set(thisAction.getId(), distr);
	        }	
	        rewarded = false;
		}
        action = findAction(sampler.sample());

		action.select();
		return action;
	}
	
	@Override
	public void setReward(String actionId, double reward) {
		rewardStats.get(actionId).add(reward);
		rewarded = true;
		findAction(actionId).reward(reward);
	}


	@Override
	public void buildModel(String model) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public String[] getModel() {
		// TODO Auto-generated method stub
		return null;
	}

}
