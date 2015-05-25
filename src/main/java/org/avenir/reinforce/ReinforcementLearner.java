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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chombo.util.AverageValue;
import org.chombo.util.ConfigUtility;
import org.chombo.util.SimpleStat;


/**
 * This interface for all reinforcement learners
 * @author pranab
 *
 */
public abstract class ReinforcementLearner {
	protected List<Action> actions = new ArrayList<Action>();
	protected int batchSize = 1;
	protected Action[] selActions;
	protected long totalTrialCount;
	protected int minTrial;
	protected Map<String, AverageValue> rewardStats = new HashMap<String, AverageValue>();
	protected boolean rewarded;
	protected int rewardScale;


	/**
	 * sets actions
	 * @param actions
	 */
	public ReinforcementLearner withActions(String[] actionIds){
		for (String actionId : actionIds) {
			actions.add(new Action(actionId));
		}
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
		selActions = new Action[batchSize];
	}

	/**
	 * @param config
	 */
	public  void initialize(Map<String, Object> config) {
		minTrial = ConfigUtility.getInt(config, "min.trial",  -1);
		batchSize = ConfigUtility.getInt(config, "batch.size",  1);
		rewardScale = ConfigUtility.getInt(config, "reward.scale",  1);
		initSelectedActions();
	}
	
	/**
	 * Selects the next action 
	 * @param roundNum
	 * @return actionID
	 */
	public abstract Action[] nextActions();

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
	
	/**
	 * @param id
	 * @return
	 */
	public Action findAction(String id) {
		Action action = null;
		for (Action thisAction : actions) {
			if (thisAction.getId().equals(id)) {
				action = thisAction;
				break;
			}
		}
		return action;
	}
	
	/**
	 * @return
	 */
	public Action findActionWithMinTrial() {
		long minTrial = Long.MAX_VALUE;
		Action action = null;
		for (Action thisAction : actions) {
			if (thisAction.getTrialCount() < minTrial) {
				minTrial = thisAction.getTrialCount();
				action = thisAction;
			}
		}
		
		return action;
	}
	
	/**
	 * @return
	 */
	public Action selectActionBasedOnMinTrial() {
		//check for min trial requirement
		Action action = null;
		if (minTrial > 0) {
			action = findActionWithMinTrial();
			if (action.getTrialCount() > minTrial) {
				action = null;
			}
		}
		return action;
	}
	
	/**
	 * @return
	 */
	public Action findBestAction() {
		String actionId = null;
		double maxReward = -1.0;
		for (String thisActionId : rewardStats.keySet()) {
			if (rewardStats.get(thisActionId).getAvgValue() > maxReward) {
				actionId = thisActionId;
			}
		}
		return findAction(actionId);
	}
}
