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
import org.chombo.util.BasicUtils;
import org.chombo.util.ConfigUtility;

/**
 * exp4 learner
 * @author pranab
 *
 */
public class ExponentialWeightExpertLearner extends MultiArmBanditLearner {
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
		int numExperts  = ConfigUtility.getInt(config, "num.experts");
 		numActions = actions.size();
        
        //flattened expert preference matrix
        double[] experts = (double[])config.get("experts");
        if (experts.length != numExperts * numActions) {
        	throw new IllegalStateException("invalid expert prefrence matrix size");
        }
        
        //build preference matrix
        this.experts = new double[numExperts][numActions];
        int r = 0;
        int c = 0;
        for (double exp : experts) {
        	this.experts[r][c++] = exp;
        	
        	//next row
        	if (c % numActions == 0) {
        		++r;
        		c = 0;
        	}
        }
        
        //initial expert weights
        expertWeights = new double[numExperts];
        for (int i = 0; i < numExperts; ++i) {
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
        	
        	double prob = (1.0 - distrConstant) * sum + distrConstant / numActions;
        	sampler.add(thisAction.getId(), prob);
        	++j;
        }
	}
	
	@Override
	public void setReward(String actionId, double reward) {
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


	@Override
	public void buildModel(String model) {
		String[] items = model.split(delim, -1);
		if (items[0].equals("action")) {
			String actionId = items[1];
			double pr = Double.parseDouble(items[2]);
			sampler.add(actionId, pr);
		} else {
			int expertIndex = Integer.parseInt(items[1]);
			double weight = Double.parseDouble(items[2]);
			expertWeights[expertIndex] = weight;
		}
	}


	@Override
	public String[] getModel() {
		String[] model = new String[actions.size() + expertWeights.length];
		int i = 0;
		
		//each actioonID and distribution
		for (Action action : actions) {
			String actionID = action.getId();
			model[i++] = "action" + delim + actionID + delim + BasicUtils.formatDouble(sampler.get(actionID), 6);
		}
		
		//weight index and distribution
		for (int j = 0; j < expertWeights.length; ++j) {
			model[i++] = "weight" + delim + i + delim + BasicUtils.formatDouble(expertWeights[j], 6);
		}
		return model;
	}


}
