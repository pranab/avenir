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
 * SoftMax reinforcement learner
 * @author pranab
 *
 */
public class SoftMaxLearner extends MultiArmBanditLearner {
	private Map<String, Double> expDistr = new HashMap<String, Double>();
	private double tempConstant;
	private double minTempConstant;
	private CategoricalSampler sampler = new CategoricalSampler();
	private String tempRedAlgorithm;
	private static final String TEMP_RED_LINEAR = "linear";
	private static final String TEMP_RED_LOG_LINEAR = "logLinear";
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		tempConstant  = ConfigUtility.getDouble(config, "temp.constant", 100.0);
		minTempConstant  = ConfigUtility.getDouble(config, "min.temp.constant", -1.0);
	    tempRedAlgorithm = ConfigUtility.getString(config,"temp.reduction.algorithm", TEMP_RED_LINEAR );
		populateMeanRewardStats();
 	}

	@Override
	public Action[] nextActions() {
		for (int i = 0; i < batchSize; ++i) {
			selActions[i] = nextAction();
		}
		return selActions;
	}

	/**
	 * @param roundNum
	 * @return
	 */
	public Action nextAction() {
		double curProb = 0.0;
		Action action = null;
		++totalTrialCount;
		
		//check for min trial requirement
		action = selectActionBasedOnMinTrial();

		if (null == action) {
			if (rewarded) {
				sampler.initialize();
				expDistr.clear();
				
				//all exp distributions 
				double sum = 0;
	            for (Action thisAction : actions) {
	            	double thisReward = meanRewardStats.get(thisAction.getId()).getMean();
	            	double distr = Math.exp(thisReward / tempConstant);
	            	expDistr.put(thisAction.getId(), distr);
	            	sum += distr;
	            }	
				
	            //prob distributions
	            for (Action thisAction : actions) {
	            	double distr = expDistr.get(thisAction.getId()) / sum;
	            	sampler.add(thisAction.getId(), distr);
	            }	
	            rewarded = false;
			}
            action = findAction(sampler.sample());
            
            //reduce constant
            long softMaxRound = totalTrialCount - minTrial;
            if (softMaxRound > 1) {
	            if (tempRedAlgorithm.equals(TEMP_RED_LINEAR)) {
	            	tempConstant /= softMaxRound;
	            } else if (tempRedAlgorithm.equals(TEMP_RED_LOG_LINEAR)) {
	            	tempConstant *= Math.log(softMaxRound) / softMaxRound;
	            }
	            
	            //apply lower bound
	            if (minTempConstant > 0 && tempConstant < minTempConstant) {
	            	tempConstant = minTempConstant;
	            }
            }            
		}
		
		action.select();
		return action;
	}
	
	@Override
	public void setReward(String action, double reward) {
		meanRewardStats.get(action).add(reward);
		findAction(action).reward(reward);
		rewarded = true;
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
