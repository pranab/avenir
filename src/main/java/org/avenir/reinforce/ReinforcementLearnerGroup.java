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

import org.chombo.util.ConfigUtility;

/**
 * @author pranab
 *
 */
public class ReinforcementLearnerGroup {
	private Map<String, ReinforcementLearner> learners = new HashMap<String, ReinforcementLearner>();
	private Map<String, Object> config;
	private String learnerType;
	private String[] actions;

	public ReinforcementLearnerGroup(Map<String, Object> config) {
		super();
		this.config = config;
		learnerType  = ConfigUtility.getString(config, "learner.type", "randomGreedy");
		actions = ConfigUtility.getString(config, "action.list").split(",");
	}
	
	/**
	 * @param learnerId
	 */
	public void addLearner(String learnerId) {
		ReinforcementLearner learner = ReinforcementLearnerFactory.create(learnerType, actions, config);
		learners.put(learnerId, learner);
	}
	
	/**
	 * @param learnerId
	 * @return
	 */
	public ReinforcementLearner getLearner(String learnerId) {
		return learners.get(learnerId);
	}	
	
	/**
	 * @param learnerId
	 * @return
	 */
	public  Action[] nextActions(String learnerId) {
		return learners.get(learnerId).nextActions();
	}
	
	/**
	 * @param learnerId
	 * @param action
	 * @param reward
	 */
	public  void setReward(String learnerId,String action, int reward) {
		learners.get(learnerId).setReward(action, reward);
	}
}
