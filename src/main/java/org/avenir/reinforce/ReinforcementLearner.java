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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chombo.stats.SimpleStat;
import org.chombo.util.ConfigUtility;


/**
 * This interface for all reinforcement learners
 * @author pranab
 *
 */
public abstract class ReinforcementLearner implements Serializable {
	protected List<Action> actions = new ArrayList<Action>();
	protected int batchSize = 1;
	protected int roundNum;
	protected Action[] selActions;
	protected int totalTrialCount;
	protected int minTrial;
	protected Map<String, SimpleStat> rewardStats = new HashMap<String, SimpleStat>();
	protected boolean rewarded;
	protected int rewardScale;
	protected boolean batchLearning;


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
	
	/**
	 * @return
	 */
	public ReinforcementLearner withBatchLearning() {
		batchLearning = true;
		return this;
	}
	
	protected void initSelectedActions() {
		selActions = new Action[batchSize];
	}

	/**
	 * @param config
	 */
	public void initialize(Map<String, Object> config) {
		minTrial = ConfigUtility.getInt(config, "min.trial",  -1);
		batchSize = ConfigUtility.getInt(config, "decision.batch.size",  1);
		rewardScale = ConfigUtility.getInt(config, "reward.scale",  1);
		roundNum = ConfigUtility.getInt(config, "current.round.num",  1);
			
		//all trials whether reward received or nor
		totalTrialCount = (roundNum - 1) * batchSize;
		initSelectedActions();
	}
	
	/**
	 * @param that
	 */
	public void merge(ReinforcementLearner that) {
		for (String actionId : that.rewardStats.keySet()) {
			rewardStats.put(actionId, that.rewardStats.get(actionId));
		}
		
		for (Action thisAction : actions) {
			int trialCount = rewardStats.get(thisAction.getId()).getCount();
			thisAction.setTrialCount(trialCount);
		}
	}
	
	/**
	 * Selects the next action 
	 * @param roundNum
	 * @return actionID
	 */
	public  Action[] nextActions() {
		for (int i = 0; i < batchSize; ++i) {
			selActions[i] = nextAction();
		}
		return selActions;
	}
	
	public abstract Action nextAction();

	/**
	 * online incremental learning
	 * @param action
	 * @param reward
	 */
	public abstract void setReward(String action, int reward);
	
	/**
	 * batch learning
	 * @param actionId
	 * @param rewardAv
	 * @param rewardStdDev
	 * @param count
	 */
	public void setReward(String actionId, double rewardAv, double rewardStdDev, int count) {
		rewardStats.get(actionId).setStats(count, rewardAv, rewardStdDev);
		
		//set trial count in action
		Action action = findAction(actionId);
		action.setTrialCount(count);
	}
	
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
			if (rewardStats.get(thisActionId).getMean() > maxReward) {
				actionId = thisActionId;
			}
		}
		return findAction(actionId);
	}
	
	/**
	 * @return
	 */
	public boolean isBatchLearning() {
		return batchLearning;
	}
	
	public int getTrialCount() {
		
		return 0;
	}
}
