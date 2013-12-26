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
import org.chombo.util.Utility;


/**
 * Interval estimator reinforcement learner based on confidence bound
 * @author pranab
 *
 */
public class IntervalEstimator extends ReinforcementLearner{
	private int binWidth;
	private int confidenceLimit;
	private int curConfidenceLimit;
	private int confidenceLimitReductionStep;
	private int confidenceLimitReductionRoundInterval;
	private int minDistrSample;
	private Map<String, HistogramStat> rewardDistr = new HashMap<String, HistogramStat>(); 
	private String[] selActions;
	private int lastRoundNum;
	
	@Override
	public void initialize(Map<String, Object> config) {
		binWidth = Utility.getInt(config, "bin.width");
		confidenceLimit = Utility.getInt(config, "confidence.limit");
		curConfidenceLimit = confidenceLimit;
		confidenceLimitReductionStep = Utility.getInt(config, "confidence.limit.reduction.step");
		confidenceLimitReductionRoundInterval = Utility.getInt(config, "confidence.limit.reduction.round.interval");
		minDistrSample = Utility.getInt(config, "min.distr.sample");
		
		for (String action : actions) {
			rewardDistr.put(action, new HistogramStat(binWidth));
		}
		
		if (batchSize == 0) {
			selActions = new String[1];
		} else {
			selActions = new String[batchSize];
		}
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

	private void adjustConfLimit(int roundNum) {
		int redStep = (roundNum - lastRoundNum) / confidenceLimitReductionRoundInterval;
		if (redStep > 0) {
			curConfidenceLimit -= confidenceLimitReductionStep;
		}
		lastRoundNum = roundNum;
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
