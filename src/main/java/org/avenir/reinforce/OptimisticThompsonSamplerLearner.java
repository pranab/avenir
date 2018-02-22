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

import org.chombo.stats.HistogramUtility;
import org.chombo.stats.NonParametricDistrRejectionSampler;
import org.chombo.util.IntRange;

/**
 * Optimistic sampson sampler
 * @author pranab
 *
 */
public class OptimisticThompsonSamplerLearner extends ThompsonSamplerLearner {
	private Map<String, Integer> meanRewards = new HashMap<String, Integer>();
	private boolean meanRewardCalculated;
	
	/**
	 * 
	 */
	private void computeRewardMean() {
		for (String actionID : nonParamDistr.keySet()) {
			computeRewardMean(actionID);
		}
		meanRewardCalculated = true;
	}

	/**
	 * @param actionID
	 */
	private void computeRewardMean(String actionID) {
		NonParametricDistrRejectionSampler<IntRange> distr = nonParamDistr.get(actionID);
		int mean = HistogramUtility.findMean(distr);
		meanRewards.put(actionID, mean);
	}

	/* (non-Javadoc)
	 * @see org.avenir.reinforce.SampsonSamplerLearner#buildModel(java.lang.String)
	 */
	@Override
	public void buildModel(String model) {
		super.buildModel(model);
		meanRewardCalculated = false;
	}	
	
	/* (non-Javadoc)
	 * @see org.avenir.reinforce.SampsonSamplerLearner#setReward(java.lang.String, double)
	 */
	@Override
	public void setReward(String actionID, double reward) {
		super.setReward(actionID, reward);
		meanRewardCalculated = false;
	}	
	
	/* (non-Javadoc)
	 * @see org.avenir.reinforce.SampsonSamplerLearner#nextAction()
	 */
	@Override
	public Action nextAction() {
		if (!meanRewardCalculated) {
			computeRewardMean();
		}
		return super.nextAction();
	}
	
	/* (non-Javadoc)
	 * @see org.avenir.reinforce.SampsonSamplerLearner#enforce(java.lang.String, int)
	 */
	@Override
	public int enforce(String actionID, int reward) {
		int meanReward = meanRewards.get(actionID);
		return reward > meanReward ? reward : meanReward;
	}
	
}
