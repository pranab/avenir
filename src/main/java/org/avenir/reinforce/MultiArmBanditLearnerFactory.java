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

import java.io.Serializable;
import java.util.Map;

/**
 * Factory to create multi arm bandit learner
 * @author pranab
 *
 */
public class MultiArmBanditLearnerFactory implements Serializable {
	public static final String INTERVAL_ESTIMATOR = "intervalEstimator";
	public static final String THOMPSON_SAMPLER = "thompsonSampler";
	public static final String OPTIMISTIC_THOMPSON_SAMPLER = "optimisticThompsonSampler";
	public static final String RANDOM_GREEDY = "randomGreedy";
	public static final String UPPER_CONFIDENCE_BOUND_ONE = "upperConfidenceBoundOne";
	public static final String UPPER_CONFIDENCE_BOUND_TWO = "upperConfidenceBoundTwo";
	public static final String SOFT_MAX = "softMax";
	public static final String ACTION_PURSUIT = "actionPursuit";
	public static final String REAWRD_COMPARISON = "rewardComparison";
	public static final String EXPONENTIAL_WEIGHT = "exponentialWeight";
	public static final String EXPONENTIAL_WEIGHT_EXPERT = "exponentialWeightExpert";
	
	
	/**
	 * @param learnerID
	 * @param actions
	 * @param config
	 * @return
	 */
	public static MultiArmBanditLearner create(String learnerType, String[] actions, Map<String, Object> config) {
		MultiArmBanditLearner learner = null;
		if (learnerType.equals(INTERVAL_ESTIMATOR)) {
			learner = new IntervalEstimatorLearner();
		} else if (learnerType.equals(THOMPSON_SAMPLER)) {
			learner = new ThompsonSamplerLearner();
		} else if (learnerType.equals(OPTIMISTIC_THOMPSON_SAMPLER)) {
			learner = new OptimisticThompsonSamplerLearner();
		} else if (learnerType.equals(RANDOM_GREEDY)) {
			learner = new RandomGreedyLearner();
		} else if (learnerType.equals(UPPER_CONFIDENCE_BOUND_ONE)) {
			learner = new UpperConfidenceBoundOneLearner();
		} else if (learnerType.equals(UPPER_CONFIDENCE_BOUND_TWO)) {
			learner = new UpperConfidenceBoundTwoLearner();
		} else if (learnerType.equals(SOFT_MAX)) {
			learner = new SoftMaxLearner();
		} else if (learnerType.equals(ACTION_PURSUIT)) {
			learner = new ActionPursuitLearner();
		} else if (learnerType.equals(REAWRD_COMPARISON)) {
			learner = new RewardComparisonLearner();
		} else if (learnerType.equals(EXPONENTIAL_WEIGHT)) {
			learner = new ExponentialWeightLearner();
		} else if (learnerType.equals(EXPONENTIAL_WEIGHT_EXPERT)) {
			learner = new ExponentialWeightExpertLearner();
		} else {
			throw new IllegalArgumentException("invalid MAB learner type:" + learnerType);
		}
		
		learner.withActions(actions).initialize(config);
		return learner;
	}
}
