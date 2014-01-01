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

import org.chombo.util.HistogramStat;
import org.chombo.util.ConfigUtility;


/**
 * Interval estimator reinforcement learner based on confidence bound
 * @author pranab
 *
 */
public class IntervalEstimator extends ReinforcementLearner{
	private int binWidth;
	private int confidenceLimit;
	private int minConfidenceLimit;
	private int curConfidenceLimit;
	private int confidenceLimitReductionStep;
	private int confidenceLimitReductionRoundInterval;
	private int minDistrSample;
	private Map<String, HistogramStat> rewardDistr = new HashMap<String, HistogramStat>(); 
	private int lastRoundNum;
	
	@Override
	public void initialize(Map<String, Object> config) {
		binWidth = ConfigUtility.getInt(config, "bin.width");
		confidenceLimit = ConfigUtility.getInt(config, "confidence.limit");
		minConfidenceLimit = ConfigUtility.getInt(config, "min.confidence.limit");
		curConfidenceLimit = confidenceLimit;
		confidenceLimitReductionStep = ConfigUtility.getInt(config, "confidence.limit.reduction.step");
		confidenceLimitReductionRoundInterval = ConfigUtility.getInt(config, "confidence.limit.reduction.round.interval");
		minDistrSample = ConfigUtility.getInt(config, "min.reward.distr.sample");
		
		for (String action : actions) {
			rewardDistr.put(action, new HistogramStat(binWidth));
		}
		
		initSelectedActions();
	}

	@Override
	public String[] nextActions(int roundNum) {
		String selAction = null;
		
		//make sure reward distributions have enough sample
		boolean lowSample = false;
		for (String action : rewardDistr.keySet()) {
			if (rewardDistr.get(action).getCount() < minDistrSample) {
				lowSample = true;
				break;
			}
		}
		
		if (lowSample) {
			//select randomly
			selAction = actions[(int)(Math.random() * actions.length)];
		} else {
			//reduce confidence limit
			adjustConfLimit(roundNum);
			
			//select as per interval estimate, choosing distr with max upper conf bound
			int maxUpperConfBound = 0;
			for (String action : rewardDistr.keySet()) {
				HistogramStat stat = rewardDistr.get(action);
				int[] confBounds = stat.getConfidenceBounds(curConfidenceLimit);
				if (confBounds[1] > maxUpperConfBound) {
					maxUpperConfBound = confBounds[1];
					selAction = action;
				}
			}
		}
		selActions[0] = selAction;
		return selActions;
	}

	/**
	 * @param roundNum
	 */
	private void adjustConfLimit(int roundNum) {
		if (curConfidenceLimit > minConfidenceLimit) {
			int redStep = (roundNum - lastRoundNum) / confidenceLimitReductionRoundInterval;
			if (redStep > 0) {
				curConfidenceLimit -= confidenceLimitReductionStep;
				if (curConfidenceLimit > minConfidenceLimit) {
					curConfidenceLimit = minConfidenceLimit;
				}
			}
			lastRoundNum = roundNum;
		}
	}
	
	@Override
	public void setReward(String action, int reward) {
		HistogramStat stat = rewardDistr.get(action);
		if (null == stat) {
			throw new IllegalArgumentException("invalid action:" + action);
		}
		stat.add(reward);
	}

}
