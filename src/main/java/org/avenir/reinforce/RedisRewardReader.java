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

import org.chombo.util.ConfigUtility;
import org.chombo.util.Pair;

import redis.clients.jedis.Jedis;

public class RedisRewardReader implements RewardReader {
	private Jedis jedis;
	private String rewardQueue;
	private long lastMessageSeq;
	private static final String NIL = "nil";
	private static final String FIELD_DELIM  =  ",";

	@Override
	public void intialize(Map stormConf) {
		//action output queue		
		String redisHost = ConfigUtility.getString(stormConf, "redis.server.host");
		int redisPort = ConfigUtility.getInt(stormConf,"redis.server.port");
		jedis = new Jedis(redisHost, redisPort);
		rewardQueue = ConfigUtility.getString(stormConf, "redis.reward.queue");
		
	}

	@Override
	public List<Pair<String, Integer>> readRewards() {
		List<Pair<String, Integer>> rewards = new ArrayList<Pair<String, Integer>>();
		List<String> messages = new ArrayList<String>();
		long latestMessageSeq = 0;
		
		while (true) {
			String message  = jedis.rpop(rewardQueue);		
			if(null != message  && !message.equals(NIL)) {
				String[] items =  message.split(",");
				long messageSeq = Long.parseLong(items[2]);
				if (messageSeq <= lastMessageSeq) {
					//already read, put back
					messages.add(message);
				} else {
					Pair<String, Integer> reward = new Pair<String, Integer>(items[0], Integer.parseInt(items[1]));
					rewards.add(reward);
					if (messageSeq > latestMessageSeq) {
						latestMessageSeq = messageSeq;
					}
					
					//decrement subscriber count and put back
					int  messageSubsCount = Integer.parseInt(items[3]);
					--messageSubsCount;
					if (messageSubsCount  > 0) {
						message = items[0] + FIELD_DELIM + items[1] + FIELD_DELIM + items[2] + FIELD_DELIM  +  messageSubsCount;
						messages.add(message);
					}
				}
			} else {
				break;
			}
		}
		
		//put messages back
		for (int i =  messages.size() -1 ; i >= 0; --i) {
			jedis.rpush(rewardQueue, messages.get(i));
		}
		
		//update latest sequence
		if (rewards.size() > 0) {
			lastMessageSeq = latestMessageSeq;
		}
		return rewards;
	}

}
