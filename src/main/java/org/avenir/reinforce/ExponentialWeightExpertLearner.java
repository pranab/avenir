
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
import org.chombo.util.ConfigUtility;

/**
 * exp4 learner
 * @author pranab
 *
 */
public class ExponentialWeightExpertLearner extends ReinforcementLearner {
	private double[] expertWeights;
	private CategoricalSampler sampler = new CategoricalSampler();
	private double distrConstant;
	private double[][] experts;
	private int numActions;
	private int numExperts;
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		distrConstant  = ConfigUtility.getDouble(config, "distr.constant", 0.1);
 		numActions = actions.size();
        
        //expert actions distributions
        Map<String,double[]> experts = (Map<String,double[]>)config.get("experts");
        numExperts = experts.size();
        this.experts = new double[numExperts][numActions];
        int i = 0;
        for (String id : experts.keySet()) {
        	this.experts[i++] = experts.get(id);
        }
        
        //weights
        expertWeights = new double[numExperts];
        for (i = 0; i < numExperts; ++i) {
        	expertWeights[i] = 1;
        }
        
        //action distributions
        updateActionDistr();
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
			updateActionDistr();
			rewarded = false;
		}
		action = findAction(sampler.sample());
		action.select();
		return action;
	}
	
	/**
	 * 
	 */
	private void updateActionDistr() {
		double sumWt = 0;
		for (double weight : expertWeights) {
			sumWt += weight;
		}
		sampler.initialize();
		
		int j = 0;
        for (Action thisAction : actions) {
        	//sum across experts for this action
        	double sum = 0;
        	for (int i = 0; i < numExperts; ++i) {
        		sum += expertWeights[i] * experts[i][j] / sumWt;
        	}
        	
        	double prob = (1.0 - distrConstant) * sum + distrConstant / actions.size();
        	sampler.add(thisAction.getId(), prob);
        	++j;
        }
	}
	
	@Override
	public void setReward(String actionId, int reward) {
		findAction(actionId).reward(reward);
		
		//update weights for experts
		double scaledReward = (double)reward / rewardScale; 
		double estReward = scaledReward / sampler.get(actionId);
		int j = findActionIndex(actionId);
    	for (int i = 0; i < numExperts; ++i) {
    		double gain = experts[i][j] * estReward;
    		double weight = expertWeights[i];
    		weight *= Math.exp(distrConstant * gain / numActions);
    		expertWeights[i] = weight;		
    	}
		rewarded = true;
	}


}
