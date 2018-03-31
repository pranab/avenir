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

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.chombo.stats.HistogramStat;
import org.chombo.util.ConfigUtility;
import org.chombo.util.Utility;


/**
 * Interval estimator reinforcement learner based on confidence bound
 * @author pranab
 *
 */
public class IntervalEstimatorLearner extends MultiArmBanditLearner{
	private double binWidth;
	private double confidenceLimit;
	private double minConfidenceLimit;
	private double curConfidenceLimit;
	private double confidenceLimitReductionStep;
	private double confidenceLimitReductionRoundInterval;
	private int minDistrSample;
	private Map<String, HistogramStat> rewardDistr = new HashMap<String, HistogramStat>(); 
	private long lastRoundNum = 1;
	private long randomSelectCount;
	private long intvEstSelectCount;
	private boolean debugOn;
	private long logCounter;
	private boolean lowSample = true;
	private static final Logger LOG = Logger.getLogger(IntervalEstimatorLearner.class);
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		binWidth = ConfigUtility.getDouble(config, "bin.width");
		confidenceLimit = ConfigUtility.getDouble(config, "confidence.limit");
		minConfidenceLimit = ConfigUtility.getDouble(config, "min.confidence.limit");
		curConfidenceLimit = confidenceLimit;
		confidenceLimitReductionStep = ConfigUtility.getDouble(config, "confidence.limit.reduction.step");
		confidenceLimitReductionRoundInterval = ConfigUtility.getDouble(config, "confidence.limit.reduction.round.interval");
		minDistrSample = ConfigUtility.getInt(config, "min.reward.distr.sample");
		
		for (Action action : actions) {
			rewardDistr.put(action.getId(), new HistogramStat(binWidth));
		}
		
		debugOn = ConfigUtility.getBoolean(config,"debug.on", false);
		if (debugOn) {
			LOG.setLevel(Level.INFO);
			LOG.info("confidenceLimit:" + confidenceLimit + " minConfidenceLimit:" + minConfidenceLimit + 
					" confidenceLimitReductionStep:" + confidenceLimitReductionStep + " confidenceLimitReductionRoundInterval:" +
					confidenceLimitReductionRoundInterval + " minDistrSample:" + minDistrSample);
		}
	}

	/* (non-Javadoc)
	 * @see org.avenir.reinforce.ReinforcementLearner#nextAction()
	 */
	@Override	
	public Action nextAction() {
		Action selAction = null;
		++logCounter;
		++totalTrialCount;
		
		//make sure reward distributions have enough sample
		if (lowSample) {
			lowSample = false;
			for (String action : rewardDistr.keySet()) {
				int sampleCount = rewardDistr.get(action).getCount();
				if (debugOn && logCounter % 100 == 0) {
					LOG.info("action:" + action + " distr sampleCount: " + sampleCount);
				}
				if (sampleCount < minDistrSample) {
					lowSample = true;
					break;
				}
			}
			
			if (!lowSample && debugOn) {
				LOG.info("got full sample");
				lastRoundNum = totalTrialCount;
			}
		}
		
		if (lowSample) {
			//select randomly
			selAction = Utility.selectRandom(actions);
			++randomSelectCount;
		} else {
			//reduce confidence limit
			adjustConfLimit();
			
			//select as per interval estimate, choosing distr with max upper conf bound
			double maxUpperConfBound = 0;
			String selActionId = null;
			for (String action : rewardDistr.keySet()) {
				HistogramStat stat = rewardDistr.get(action);
				double[] confBounds = stat.getConfidenceBounds(curConfidenceLimit);
				if (debugOn) {
					LOG.info("curConfidenceLimit:" + curConfidenceLimit + " action:" + action + " conf bounds:" + confBounds[0] + "  " + confBounds[1]);
				}
				if (confBounds[1] > maxUpperConfBound) {
					maxUpperConfBound = confBounds[1];
					selActionId = action;
				}
			}
			selAction = findAction(selActionId);
			++intvEstSelectCount;
		}
		selAction.select();
		return selAction;
	}

	/**
	 * @param roundNum
	 */
	private void adjustConfLimit() {
		if (curConfidenceLimit > minConfidenceLimit) {
			int redStep = (int)((totalTrialCount - lastRoundNum) / confidenceLimitReductionRoundInterval);
			if (debugOn) {
				LOG.info("redStep:" +  redStep + " roundNum:"  + totalTrialCount + " lastRoundNum:" + lastRoundNum);
			}
			if (redStep > 0) {
				curConfidenceLimit -=  (redStep * confidenceLimitReductionStep);
				if (curConfidenceLimit < minConfidenceLimit) {
					curConfidenceLimit = minConfidenceLimit;
				}
				if (debugOn) {
					LOG.info("reduce conf limit roundNum:" +  totalTrialCount + " lastRoundNum:"  + lastRoundNum);
				}
				lastRoundNum = totalTrialCount;
			}
		}
	}
	
	@Override
	public void setReward(String action, double reward) {
		HistogramStat stat = rewardDistr.get(action);
		if (null == stat) {
			throw new IllegalArgumentException("invalid action:" + action);
		}
		stat.add(reward);
		findAction(action).reward(reward);
		if (debugOn) {
			LOG.info("setReward action:" + action + " reward:" + reward + " sample count:" + stat.getCount());
		}
	}

	public String getStat() {
		return "randomSelectCount:" + randomSelectCount + " intvEstSelectCount:" + intvEstSelectCount; 
	}

	@Override
	public void buildModel(String model) {
		String[] items = model.split(delim, -1);
		String actionId = items[0];
		HistogramStat stat = rewardDistr.get(actionId);
		stat.initializeBins(items, 1);
		
		//update action state
		Action action = findAction(actionId);
		action.setTrialCount(stat.getCount());
		action.setTotalReward(stat.getSum());
	}

	@Override
	public String[] getModel() {
		String[] model = new String[actions.size()];
		int i = 0;
		for (String actionId : rewardDistr.keySet()) {
			HistogramStat stat = rewardDistr.get(actionId);
			stat.withSerializeBins(true);
			model[i++] = actionId + delim + stat.toString();
		}
		return model;
	}
}
