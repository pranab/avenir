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

import org.chombo.stats.NonParametricDistrRejectionSampler;
import org.chombo.util.ConfigUtility;
import org.chombo.util.IntRange;
import org.chombo.util.Record;

/**
 * Sampson sampler probabilistic matching reinforcement learning
 * @author pranab
 *
 */
public class ThompsonSamplerLearner extends MultiArmBanditLearner {
	protected Map<String, NonParametricDistrRejectionSampler<IntRange>> nonParamDistr = 
			new HashMap<String, NonParametricDistrRejectionSampler<IntRange>>();
	protected Map<String, Integer> trialCounts = new HashMap<String, Integer>();
	private int minSampleSize;
	private int maxReward;
	private int binWidth;
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		minSampleSize = ConfigUtility.getInt(config, "min.sample.size");
		maxReward = ConfigUtility.getInt(config, "max.reward");
		binWidth = ConfigUtility.getInt(config, "bin.width");
	}
	
	@Override
	public void merge(MultiArmBanditLearner that) {
		ThompsonSamplerLearner thatLearner = (ThompsonSamplerLearner)that;
		for (Action action : actions) {
			String actionId = action.getId();
			NonParametricDistrRejectionSampler<IntRange> thisDistr = nonParamDistr.get(actionId);
			NonParametricDistrRejectionSampler<IntRange> thatDistr = thatLearner.nonParamDistr.get(actionId);
			if (null != thatDistr) {
				if (null == thisDistr) {
					thisDistr = new NonParametricDistrRejectionSampler<IntRange>();
					nonParamDistr.put(actionId, thisDistr);
				}
				thisDistr.merge(thatDistr);
			}
			
			Integer count = trialCounts.get(actionId);
			Integer thatCount = thatLearner.trialCounts.get(actionId);
			int aggrCount = (null == count ? 0 : count) + (null == thatCount ? 0 : thatCount);
			trialCounts.put(actionId, aggrCount);
		}
	}
	
	/* (non-Javadoc)
	 * @see org.avenir.reinforce.MultiArmBanditLearner#setReward(java.lang.String, double)
	 */
	@Override
	public void setReward(String actionID, double reward) {
		NonParametricDistrRejectionSampler<IntRange> distr = nonParamDistr.get(actionID);
		int binIndex = (int)(reward / binWidth);
		int binBeg = binIndex * binWidth;
		int binEnd = binBeg + binWidth - 1;
		IntRange range = new IntRange(binBeg, binEnd);
		distr.add(range);
		trialCounts.put(actionID, trialCounts.get(actionID) + 1);
		
		findAction(actionID).reward(reward);
	}
	
	/* (non-Javadoc)
	 * @see org.avenir.reinforce.MultiArmBanditLearner#nextAction()
	 */
	@Override
	public Action  nextAction() {
		String slectedActionID = null;
		int maxRewardCurrent = 0;
		int reward = 0;
		++totalTrialCount;
		
		for (String actionID : trialCounts.keySet()) {
			if (trialCounts.get(actionID) > minSampleSize) {
				IntRange range = nonParamDistr.get(actionID).sample();
				reward = (range.getLeft() + range.getRight()) / 2;
				reward = enforce(actionID, reward);
			} else {
				reward = (int)(Math.random() * maxReward);
			}
			
			if (reward > maxRewardCurrent) {
				slectedActionID= actionID;
				maxRewardCurrent = reward;
			}
		}
		
		Action selAction = findAction(slectedActionID);
		selAction.select();
		return selAction;
	}
	
	/**
	 * @param actionID
	 * @param reward
	 * @return
	 */
	public int enforce(String actionID, int reward) {
		return reward;
	}

	/* (non-Javadoc)
	 * @see org.avenir.reinforce.MultiArmBanditLearner#buildModel(java.lang.String)
	 */
	@Override
	public void buildModel(String model) {
		Record record = new Record(model, delim);
		if (record.getSize() > 1) {
			buildExistingModel(record);
		} else {
			buildInitialModel(record);
		}
	}
	
	/**
	 * @param record
	 */
	private void buildExistingModel(Record record) {
		String actionId = record.getString();
		int numBins = record.getInt();
		
		//populate distribution
		NonParametricDistrRejectionSampler<IntRange> distr = new NonParametricDistrRejectionSampler<IntRange>();
		int count = 0;
		for (int i = 0; i < numBins; ++i) {
			int binIndex = record.getInt();
			int binBeg = binIndex * binWidth;
			int binEnd = binBeg + binWidth - 1;
			IntRange range = new IntRange(binBeg, binEnd);
			int value = record.getInt();
			distr.add(range, value);
			count += value;
		}
		
		nonParamDistr.put(actionId, distr);
		trialCounts.put(actionId, count);
	}

	/**
	 * @param record
	 */
	private void buildInitialModel(Record record) {
		String actionId = record.getString();
		int numBins = maxReward / binWidth + 1;
		
		//populate distribution
		NonParametricDistrRejectionSampler<IntRange> distr = new NonParametricDistrRejectionSampler<IntRange>();
		int count = 0;
		for (int i = 0; i < numBins; ++i) {
			int binIndex = i;
			int binBeg = binIndex * binWidth;
			int binEnd = binBeg + binWidth - 1;
			IntRange range = new IntRange(binBeg, binEnd);
			int value = 1;
			distr.add(range, value);
			count += value;
		}		
		nonParamDistr.put(actionId, distr);
		trialCounts.put(actionId, count);
	}

	/* (non-Javadoc)
	 * @see org.avenir.reinforce.MultiArmBanditLearner#getModel()
	 */
	@Override
	public String[] getModel() {
		String[] model = new String[actions.size()];
		int i = 0;
		for (String actionID : nonParamDistr.keySet()) {
			Map<IntRange, Double> distr = nonParamDistr.get(actionID).getDistr();
			int numBins = distr.size();
			Record record = new Record(2 + 2 * numBins);
			record.setString(actionID);
			record.setInt(numBins);
			for (IntRange value : distr.keySet()) {
				int binIndex = value.getLeft() / binWidth;
				int distrCount = (int)Math.round(distr.get(value));
				record.setInt(binIndex);
				record.setInt(distrCount);
			}
			model[i++] = record.withDelim(delim).toString();
		}
		return model;
	}

}
