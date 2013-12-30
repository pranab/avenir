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
import java.util.List;
import java.util.Map;

import org.chombo.storm.GenericBolt;
import org.chombo.storm.MessageHolder;
import org.chombo.util.Utility;

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
	private ReinforcementLearner  learner = null;
	private Jedis jedis;
	private String actionQueue;
	private ActionWriter  actionWriter;;
	
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
		String learnerType = Utility.getString(stormConf, "reinforcement.lrearner.type");
		String[] actions = Utility.getString(stormConf, "reinforcement.lrearneractions").split(",");
		Map<String, Object> typedConf = Utility.toTypedMap(stormConf);
		learner =  ReinforcementLearnerFactory.create(learnerType, actions, typedConf);
		
		//action output queue		
		if (Utility.getString(stormConf, "reinforcement.lrearner.action.writer").equals("redis")) {
			actionWriter = new RedisActionWriter();
			actionWriter.intialize(stormConf);
		}
		
	}

	/* (non-Javadoc)
	 * @see org.chombo.storm.GenericBolt#process(backtype.storm.tuple.Tuple)
	 */
	@Override
	public boolean process(Tuple input) {
		if (input.contains(ROUND_NUM)) {
			//select action for next round
			String eventID = input.getStringByField(EVENT_ID);
			int roundNum = input.getIntegerByField(ROUND_NUM);
			String[] actions = learner.nextActions(roundNum);
			actionWriter.write(eventID, actions);
		} else {
			//reward feedback
			String action = input.getStringByField(ACTION_ID);
			int reward = input.getIntegerByField(REWARD);
			learner.setReward(action, reward);
		}
		return true;
	}

	@Override
	public List<MessageHolder> getOutput() {
		return null;
	}

}
