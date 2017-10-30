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
import org.chombo.util.BasicUtils;
import org.chombo.util.ConfigUtility;


/**
 * exp3 learner
 * @author pranab
 *
 */
public class ExponentialWeightLearner extends MultiArmBanditLearner {
	private Map<String, Double> weightDistr = new HashMap<String, Double>();
	private CategoricalSampler sampler = new CategoricalSampler();
	private double distrConstant;
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		distrConstant  = ConfigUtility.getDouble(config, "distr.constant", 100.0);
        
		double intialProb = 1.0  / actions.size();
        for (Action action : actions) {
        	weightDistr.put(action.getId(), 1.0);
        	sampler.add(action.getId(), intialProb);
        }
 	}
	

	/**
	 * @param roundNum
	 * @return
	 */
	@Override
	public Action nextAction() {
		Action action = null;
		++totalTrialCount;
		
		if (rewarded) {
			double sumWt = 0;
			for (String actionId : weightDistr.keySet()) {
				sumWt += weightDistr.get(actionId);
			}
			sampler.initialize();
	        for (Action thisAction : actions) {
	        	double prob = (1.0 - distrConstant) * weightDistr.get(thisAction.getId()) / sumWt + distrConstant / actions.size();
	        	sampler.add(thisAction.getId(), prob);
	        }
			rewarded = false;
		}
		action = findAction(sampler.sample());
		action.select();
		return action;
	}
	
	@Override
	public void setReward(String actionId, double reward) {
		findAction(actionId).reward(reward);
		double weight = weightDistr.get(actionId);
		double scaledReward = (double)reward / rewardScale; 
		weight *= Math.exp(distrConstant * (scaledReward / sampler.get(actionId)) / actions.size());
		weightDistr.put(actionId, weight);
		rewarded = true;
	}


	/* (non-Javadoc)
	 * @see org.avenir.reinforce.MultiArmBanditLearner#buildModel(java.lang.String)
	 */
	@Override
	public void buildModel(String model) {
		String[] items = model.split(delim, -1);
		String actionId = items[0];
		double weight = Double.parseDouble(items[1]);
		weightDistr.put(actionId, weight);
		double pr = Double.parseDouble(items[2]);
		sampler.add(actionId, pr);
	}


	@Override
	public String[] getModel() {
		String[] model = new String[actions.size()];
		int i = 0;
		//each actioonID , weight and distribution
		for (Action action : actions) {
			String actionID = action.getId();
			model[i++] = actionID + delim + BasicUtils.formatDouble(weightDistr.get(actionID), 6) +  delim + 
					BasicUtils.formatDouble(sampler.get(actionID), 6);
		}
		return model;
	}

}
