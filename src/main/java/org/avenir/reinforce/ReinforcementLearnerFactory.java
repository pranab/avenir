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

/**
 * Factory to create reinforcement learner
 * @author pranab
 *
 */
public class ReinforcementLearnerFactory {
	/**
	 * @param learnerID
	 * @param actions
	 * @param config
	 * @return
	 */
	public static ReinforcementLearner create(String learnerType, String[] actions, Map<String, Object> config) {
		ReinforcementLearner learner = null;
		if (learnerType.equals("intervalEstimator")) {
			learner = new IntervalEstimatorLearner();
		} else if (learnerType.equals("sampsonSampler")) {
			learner = new SampsonSamplerLearner();
		} else if (learnerType.equals("optimisticSampsonSampler")) {
			learner = new OptimisticSampsonSamplerLearner();
		} else if (learnerType.equals("randomGreedy")) {
			learner = new RandomGreedyLearner();
		} else if (learnerType.equals("upperConfidenceBoundOne")) {
			learner = new UpperConfidenceBoundOneLearner();
		} else if (learnerType.equals("upperConfidenceBoundTwo")) {
			learner = new UpperConfidenceBoundTwoLearner();
		} else if (learnerType.equals("softMax")) {
			learner = new SoftMaxLearner();
		} else if (learnerType.equals("actionPursuit")) {
			learner = new ActionPursuitLearner();
		} else if (learnerType.equals("rewardComparison")) {
			learner = new RewardComparisonLearner();
		} else {
			throw new IllegalArgumentException("invalid learner type:" + learnerType);
		}
		
		learner.withActions(actions).initialize(config);
		return learner;
	}
}
