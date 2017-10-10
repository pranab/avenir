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

package org.avenir.storm;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.avenir.reinforce.Action;
import org.avenir.reinforce.ActionWriter;
import org.avenir.reinforce.MultiArmBanditLearner;
import org.avenir.reinforce.MultiArmBanditLearnerFactory;
import org.avenir.reinforce.RewardReader;
import org.chombo.storm.GenericBolt;
import org.chombo.storm.MessageHolder;
import org.chombo.util.ConfigUtility;
import org.chombo.util.Pair;

import redis.clients.jedis.Jedis;
import backtype.storm.task.TopologyContext;
import backtype.storm.tuple.Tuple;

/**
 * Reinforcement learner bolt. Any RL algorithm can be used
 * @author pranab
 *
 */
public class ReinforcementLearnerBolt extends GenericBolt {
	private static final long serialVersionUID = 6746219511729480056L;
	public static final String EVENT_ID = "eventID";
	public static final String ACTION_ID = "actionID";
	public static final String ROUND_NUM = "roundNUm";
	public static final String REWARD = "reward";
	private List<MessageHolder> messages = new ArrayList<MessageHolder>();
	private MultiArmBanditLearner  learner = null;
	private Jedis jedis;
	private String actionQueue;
	private ActionWriter  actionWriter;
	private RewardReader rewardReader; 
	private static final Logger LOG = Logger.getLogger(ReinforcementLearnerBolt.class);
			
			
	@Override
	public Map<String, Object> getComponentConfiguration() {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see org.chombo.storm.GenericBolt#intialize(java.util.Map, backtype.storm.task.TopologyContext)
	 */
	@Override
	public void intialize(Map stormConf, TopologyContext context) {
		//intialize learner
		String learnerType = ConfigUtility.getString(stormConf, "reinforcement.learner.type");
		String[] actions = ConfigUtility.getString(stormConf, "reinforcement.learrner.actions").split(",");
		Map<String, Object> typedConf = ConfigUtility.toTypedMap(stormConf);
		learner =  MultiArmBanditLearnerFactory.create(learnerType, actions, typedConf);
		
		//action output queue		
		if (ConfigUtility.getString(stormConf, "reinforcement.learrner.action.writer").equals("redis")) {
			actionWriter = new RedisActionWriter();
			actionWriter.intialize(stormConf);
			
			rewardReader = new RedisRewardReader();
			rewardReader.intialize(stormConf);
		}
		debugOn = ConfigUtility.getBoolean(stormConf,"debug.on", false);
		if (debugOn) {
			LOG.setLevel(Level.INFO);;
		}
		messageCountInterval = ConfigUtility.getInt(stormConf,"log.message.count.interval", 100);
		LOG.info("debugOn:" + debugOn);
	}

	/* (non-Javadoc)
	 * @see org.chombo.storm.GenericBolt#process(backtype.storm.tuple.Tuple)
	 */
	@Override
	public boolean process(Tuple input) {
		if (input.contains(ROUND_NUM)) {
			//get rewards
			List<Pair<String, Integer>> rewards =  rewardReader.readRewards();
			for (Pair<String, Integer> reward : rewards) {
				learner.setReward(reward.getLeft(), reward.getRight());
			}
			if (debugOn && rewards.size() > 0) {
				LOG.info("number of reward data:" + rewards.size() );
			}			
			
			//select action for next round
			String eventID = input.getStringByField(EVENT_ID);
			int roundNum = input.getIntegerByField(ROUND_NUM);
			Action[] actions = learner.nextActions();
			actionWriter.write(eventID, actions);
			if (debugOn) {
				if (messageCounter % messageCountInterval == 0)
					LOG.info("processed event message - message counter:" + messageCounter );
					LOG.info("learner stat:" + learner.getModel());
			}
		} else {
			//reward feedback
			String action = input.getStringByField(ACTION_ID);
			int reward = input.getIntegerByField(REWARD);
			learner.setReward(action, reward);
			if (debugOn) {
				if (messageCounter % messageCountInterval == 0)
					LOG.info("processed reward message - message counter:" + messageCounter );
			}
		}
		return true;
	}

	@Override
	public List<MessageHolder> getOutput() {
		return null;
	}

}
