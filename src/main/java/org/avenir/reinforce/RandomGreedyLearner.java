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

import org.chombo.stats.MeanStat;
import org.chombo.stats.SimpleStat;
import org.chombo.util.ConfigUtility;
import org.chombo.util.Utility;

/**
 * Random greedy reinforcement learner
 * @author pranab
 *
 */
public class RandomGreedyLearner extends MultiArmBanditLearner {
	private double  randomSelectionProb;
	private String  probRedAlgorithm;
	private  double	probReductionConstant;
	private double minProb;

	private static final String PROB_RED_NONE = "none";
	private static final String PROB_RED_LINEAR = "linear";
	private static final String PROB_RED_LOG_LINEAR = "logLinear";
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);;
		randomSelectionProb = ConfigUtility.getDouble(config, "random.selection.prob", 0.5);
	    probRedAlgorithm = ConfigUtility.getString(config,"prob.reduction.algorithm", PROB_RED_LINEAR );
        probReductionConstant = ConfigUtility.getDouble(config, "prob.reduction.constant",  1.0);
        minProb = ConfigUtility.getDouble(config, "min.prob",  -1.0);
 	}

	
	/**
	 * @param roundNum
	 * @return
	 */
	@Override
	public Action nextAction() {
		double curProb = 0.0;
		Action action = null;
		++totalTrialCount;

		//check for min trial requirement
		action = selectActionBasedOnMinTrial();

		if (null == action) {
			if (probRedAlgorithm.equals(PROB_RED_NONE )) {
				curProb = randomSelectionProb;
			} else if (probRedAlgorithm.equals(PROB_RED_LINEAR )) {
				curProb = randomSelectionProb * probReductionConstant / totalTrialCount ;
			} else if (probRedAlgorithm.equals(PROB_RED_LOG_LINEAR )){
	   			curProb = randomSelectionProb * probReductionConstant * Math.log(totalTrialCount) / totalTrialCount;
			} else {
				throw new IllegalArgumentException("Invalid probability reduction algorithms");
			}
			curProb = curProb <= randomSelectionProb ? curProb : randomSelectionProb;
			
			//non stationary reward
			if (minProb > 0 && curProb < minProb) {
				curProb = minProb;
			}
			
	       	if (curProb < Math.random()) {
	    		//select random
	    		action = Utility.selectRandom(actions);
	    	} else {
	    		//select best
	    		int bestReward = 0;
	            for (Action thisAction : actions) {
	            	int thisReward = (int)(rewardStats.get(thisAction.getId()).getMean());
	            	if (thisReward >  bestReward) {
	            		bestReward = thisReward;
	            		action = thisAction;
	            	}
	            }
	    	}
		}
		action.select();
		return action;
	}

	@Override
	public void setReward(String actionId, double reward) {
		meanRewardStats.get(actionId).add(reward);
		findAction(actionId).reward(reward);
	}

	@Override
	public void buildModel(String model) {
		String[] items = model.split(delim, -1);
		String actionId = items[0];
		int count = Integer.parseInt(items[1]);
		double sum = Double.parseDouble(items[2]);
		double mean = Double.parseDouble(items[3]);
		meanRewardStats.put(actionId, new MeanStat(count,sum,mean));
	}

	@Override
	public String[] getModel() {
		String[] model = new String[actions.size()];
		int i = 0;
		for (String actionId : meanRewardStats.keySet()) {
			model[i++] = actionId + meanRewardStats.get(actionId).toString();
		}
		return model;
	}

}
