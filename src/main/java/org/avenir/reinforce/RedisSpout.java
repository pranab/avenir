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

import org.chombo.storm.GenericSpout;
import org.chombo.storm.MessageHolder;
import org.chombo.util.ConfigUtility;

import redis.clients.jedis.Jedis;

import backtype.storm.task.TopologyContext;
import backtype.storm.tuple.Values;

public class RedisSpout extends GenericSpout {
	private static final long serialVersionUID = 4571831489023437625L;
	private Jedis jedis;
	private String eventQueue;
	private String rewardQueue;
	private MessageHolder pendingMsgHolder = null;
	private static final String NIL = "nil";
	public static final String EVENT_STREAM = "eventStream";
	public static final String REWARD_STREAM = "rewardStream";
	

	@Override
	public void close() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void activate() {
		// TODO Auto-generated method stub
	}

	@Override
	public void deactivate() {
		// TODO Auto-generated method stub
	}

	@Override
	public Map<String, Object> getComponentConfiguration() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void intialize(Map stormConf, TopologyContext context) {
		String redisHost = ConfigUtility.getString(stormConf, "redis.server.host");
		int redisPort = ConfigUtility.getInt(stormConf,"redis.server.port");
		jedis = new Jedis(redisHost, redisPort);
		eventQueue = ConfigUtility.getString(stormConf, "redis.event.queue");
		rewardQueue = ConfigUtility.getString(stormConf, "redis.reward.queue");
	}

	@Override
	public MessageHolder nextSpoutMessage() {
		MessageHolder msgHolder = null;
		if (null != pendingMsgHolder) {
			//anything pending
			msgHolder = pendingMsgHolder;
			pendingMsgHolder = null;
		} else {
			String message  = jedis.rpop(eventQueue);		
			if(null != message  && !message.equals(NIL)) {
				//message in event queue
				String[] items = message.split(",");
				Values values = new Values(items[0],items[1]);
				msgHolder = new  MessageHolder(values);
				msgHolder.setStream(EVENT_STREAM);
			} 
			
			message  = jedis.rpop(rewardQueue);
			if(null != message  && !message.equals(NIL)) {
					//message in reward queue
					String[] items = message.split(",");
					Values values = new Values(items[0],items[1]);
					if (null == msgHolder) {
						//nothing in event queue, return this message
						msgHolder = new  MessageHolder(values);
						msgHolder.setStream(REWARD_STREAM);
					} else {
						//message from event queue, make this message pending
						pendingMsgHolder = new  MessageHolder(values);
						pendingMsgHolder.setStream(REWARD_STREAM);
					}
			}
		}
		return msgHolder;
	}

	@Override
	public void handleFailedMessage(Values tuple) {
		// TODO Auto-generated method stub
		
	}

}
