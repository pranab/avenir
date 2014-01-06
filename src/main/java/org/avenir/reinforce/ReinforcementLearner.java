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
 * This interface for all reinforcement learners
 * @author pranab
 *
 */
public abstract class ReinforcementLearner {
	protected String[] actions;
	protected int batchSize;
	protected String[] selActions;

	/**
	 * sets actions
	 * @param actions
	 */
	public ReinforcementLearner withActions(String[] actions){
		this.actions = actions;
		return this;
	}
	
	/**
	 * If a batch size worth of actions need to be selected
	 * @param batchSize
	 * @return
	 */
	public ReinforcementLearner withBatchSize(int batchSize) {
		this.batchSize = batchSize;
		return this;
	}
	
	protected void initSelectedActions() {
		if (batchSize == 0) {
			selActions = new String[1];
		} else {
			selActions = new String[batchSize];
		}
		
	}

	/**
	 * @param config
	 */
	public abstract void initialize(Map<String, Object> config);
	
	/**
	 * Selects the next action 
	 * @param roundNum
	 * @return actionID
	 */
	public abstract String[] nextActions(int roundNum);

	/**
	 * @param action
	 * @param reward
	 */
	public abstract void setReward(String action, int reward);
	
	/**
	 * @return
	 */
	public  String getStat() {
		return "";
	}
	
}
