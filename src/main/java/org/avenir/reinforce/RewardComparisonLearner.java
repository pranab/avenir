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

import org.chombo.stats.CategoricalSampler;
import org.chombo.stats.SimpleStat;
import org.chombo.util.ConfigUtility;

/**
 * @author pranab
 *
 */
public class RewardComparisonLearner extends MultiArmBanditLearner {
	private double preferenceChangeRate;
	private double refRewardChangeRate;
	private double intialRefReward;
	private CategoricalSampler sampler = new CategoricalSampler();
	private Map<String, Double> actionPrefs = new HashMap<String, Double>();
	private double refReward;
	private Map<String, Double> expDistr = new HashMap<String, Double>();
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		preferenceChangeRate  = ConfigUtility.getDouble(config, "preference.change.rate", 0.01);
		refRewardChangeRate  = ConfigUtility.getDouble(config, "reference.reward.change.rate", 0.01);
		intialRefReward = ConfigUtility.getDouble(config, "intial.reference.reward", 100.0);
		refReward = intialRefReward;
		
		double intialProb = 1.0 / actions.size();
        for (Action action : actions) {
        	sampler.add(action.getId(), intialProb);
        	rewardStats.put(action.getId(), new SimpleStat());
        	actionPrefs.put(action.getId(), 0.0);
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
			sampler.initialize();
			
			//exponentials
			double expSum = 0;
	        for (Action thisAction : actions) {
	        	distr = Math.exp(actionPrefs.get(thisAction.getId()));
	        	expDistr.put(thisAction.getId(), distr);
	        	expSum += distr;
	        }
	        
	        //prob distr
            for (Action thisAction : actions) {
            	distr = expDistr.get(thisAction.getId()) / expSum;
            	sampler.add(thisAction.getId(), distr);
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
		
		//update action preference
		double meanReward = rewardStats.get(actionId).getMean();
		double actionPref = actionPrefs.get(actionId) + preferenceChangeRate * (meanReward - refReward);
		actionPrefs.put(actionId, actionPref);
		
		//update reference reward
		refReward += refRewardChangeRate * (meanReward - refReward);
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
