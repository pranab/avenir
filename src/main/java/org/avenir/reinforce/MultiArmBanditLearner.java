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

import org.chombo.stats.MeanStat;
import org.chombo.stats.SimpleStat;
import org.chombo.util.ConfigUtility;


/**
 * This interface for all reinforcement learners
 * @author pranab
 *
 */
public abstract class MultiArmBanditLearner implements Serializable {
	protected String id;
	protected List<Action> actions = new ArrayList<Action>();
	protected int batchSize = 1;
	protected int roundNum;
	protected Action[] selActions;
	protected int totalTrialCount;
	protected int minTrial;
	protected Map<String, SimpleStat> rewardStats = new HashMap<String, SimpleStat>();
	protected Map<String, MeanStat> meanRewardStats = new HashMap<String, MeanStat>();
	protected boolean rewarded;
	protected int rewardScale;
	protected boolean batchLearning;
	protected String delim = ",";


	/**
	 * sets actions
	 * @param actions
	 */
	public MultiArmBanditLearner withActions(String[] actionIds){
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
	public MultiArmBanditLearner withBatchSize(int batchSize) {
		this.batchSize = batchSize;
		return this;
	}
	
	/**
	 * @return
	 */
	public MultiArmBanditLearner withBatchLearning() {
		batchLearning = true;
		return this;
	}
	
	/**
	 * @param delim
	 * @return
	 */
	public MultiArmBanditLearner withDelim(String delim) {
		this.delim = delim;
		return this;
	}
	
	protected void initSelectedActions() {
		selActions = new Action[batchSize];
	}

	/**
	 * Initialize with parameters
	 * @param config
	 */
	public void initialize(Map<String, Object> config) {
		minTrial = ConfigUtility.getInt(config, "min.trial",  -1);
		batchSize = ConfigUtility.getInt(config, "decision.batch.size",  1);
		rewardScale = ConfigUtility.getInt(config, "reward.scale",  1);
		roundNum = ConfigUtility.getInt(config, "current.decision.round",  1);
			
		//all trials whether reward received or nor
		totalTrialCount = (roundNum - 1) * batchSize;
		initSelectedActions();
	}
	
	/**
	 * merge two learners
	 * @param that
	 */
	public void merge(MultiArmBanditLearner that) {
		for (String actionId : that.rewardStats.keySet()) {
			rewardStats.put(actionId, that.rewardStats.get(actionId));
		}
		
		for (Action thisAction : actions) {
			int trialCount = rewardStats.get(thisAction.getId()).getCount();
			thisAction.setTrialCount(trialCount);
		}
	}
	
	/**
	 * 
	 */
	protected void populateMeanRewardStats() {
        for (Action action : actions) {
        	meanRewardStats.put(action.getId(), new MeanStat());
        }
	}

	
	/**
	 * build model based current state
	 * @param model
	 */
	public abstract void buildModel(String model);
	
	/**
	 * decides the next action list
	 * @param roundNum
	 * @return actionID
	 */
	public  Action[] nextActions() {
		for (int i = 0; i < batchSize; ++i) {
			selActions[i] = nextAction();
		}
		return selActions;
	}
	
	/**
	 * decides next action
	 * @return
	 */
	public abstract Action nextAction();

	/**
	 * set reward and update model
	 * @param action
	 * @param reward
	 */
	public abstract void setReward(String action, double reward);
	
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
	 * get serialized model
	 * @return
	 */
	public abstract String[] getModel();
	
	/**
	 * find action, given ID
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
	 * find index of action
	 * @param id
	 * @return
	 */
	public int findActionIndex(String id) {
		int i = 0;
		for (Action thisAction : actions) {
			if (thisAction.getId().equals(id)) {
				break;
			}
			++i;
		}
		return i;
	}

	/**
	 * find action with trials less than threshold
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
	 * finds action based on max average reward
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
	
	/**
	 * @param model
	 */
	protected void buildMeanRewardStatModel(String model) {
		String[] items = model.split(delim, -1);
		String actionId = items[0];
		int count = Integer.parseInt(items[1]);
		double sum = Double.parseDouble(items[2]);
		double mean = Double.parseDouble(items[3]);
		
		//update stats
		meanRewardStats.put(actionId, new MeanStat(count,sum,mean));
		
		//update action state
		Action action = findAction(actionId);
		action.setTrialCount(count);
		action.setTotalReward(sum);
	}
	
	/**
	 * @return
	 */
	protected String[] getMeanRewardStatModel() {
		String[] model = new String[actions.size()];
		int i = 0;
		for (String actionId : meanRewardStats.keySet()) {
			model[i++] = actionId + delim + meanRewardStats.get(actionId).toString();
		}
		return model;
	}

}
