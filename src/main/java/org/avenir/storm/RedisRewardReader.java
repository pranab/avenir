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
import org.avenir.reinforce.RewardReader;
import org.chombo.util.ConfigUtility;
import org.chombo.util.Pair;

import redis.clients.jedis.Jedis;

public class RedisRewardReader implements RewardReader {
	private Jedis jedis;
	private String rewardQueue;
	private long startOffset = -1;
	private boolean debugOn;
	private static final String FIELD_DELIM  =  ",";
	private static final Logger LOG = Logger.getLogger(RedisRewardReader.class);
	private static final String NIL = "nil";

	@Override
	public void intialize(Map stormConf) {
		//action output queue		
		String redisHost = ConfigUtility.getString(stormConf, "redis.server.host");
		int redisPort = ConfigUtility.getInt(stormConf,"redis.server.port");
		jedis = new Jedis(redisHost, redisPort);
		rewardQueue = ConfigUtility.getString(stormConf, "redis.reward.queue");
		debugOn = ConfigUtility.getBoolean(stormConf,"debug.on", false);
		if (debugOn) {
			LOG.setLevel(Level.INFO);
		}	
	}

	@Override
	public List<Pair<String, Integer>> readRewards() {
		List<Pair<String, Integer>> rewards = new ArrayList<Pair<String, Integer>>();
		/*
		List<String> messages = jedis.lrange(rewardQueue, startOffset,  startOffset+1000000L);
		if (debugOn) {
			LOG.info("startOffset:" + startOffset + " num of reward records:" + messages.size());
		}
		for (String message : messages) {
			String[] items =  message.split(FIELD_DELIM);
			Pair<String, Integer> reward = new Pair<String, Integer>(items[0], Integer.parseInt(items[1]));
			rewards.add(reward);
			if (debugOn) {
				LOG.info("reward:" + message);
			}
		}
		startOffset += messages.size();
		*/
		
		String message = null;
		while(true) {
			message = jedis.lindex(rewardQueue, startOffset);
			if(null != message  && !message.equals(NIL)) {
				if (debugOn) {
					LOG.info("next reward:" + message + " startOffset:" + startOffset);
				}
				String[] items =  message.split(FIELD_DELIM);
				Pair<String, Integer> reward = new Pair<String, Integer>(items[0], Integer.parseInt(items[1]));
				rewards.add(reward);
				--startOffset;
			} else {
				break;
			}
		}
		return rewards;
	}

}
