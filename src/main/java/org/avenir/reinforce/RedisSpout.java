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

import redis.clients.jedis.Jedis;

import backtype.storm.task.TopologyContext;
import backtype.storm.tuple.Values;

public class RedisSpout extends GenericSpout {
	private static final long serialVersionUID = 4571831489023437625L;
	private Jedis jedis;
	private String eventQueue;
	private String rewardQueue;
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
		String redisHost = stormConf.get("redis.server.host").toString();
		int redisPort = new Integer(stormConf.get("redis.server.port").toString());
		jedis = new Jedis(redisHost, redisPort);
		eventQueue =  stormConf.get("redis.event.queue").toString();
		rewardQueue =  stormConf.get("redis.reward.queue").toString();
		
	}

	@Override
	public MessageHolder nextSpoutMessage() {
		MessageHolder msgHolder = null;
		String message  = jedis.rpop(eventQueue);		
		if(null != message  && !message.equals(NIL)) {
			String[] items = message.split(",");
			Values values = new Values(items[0],items[1]);
			msgHolder = new  MessageHolder(values);
			msgHolder.setStream(EVENT_STREAM);
		} else {
			message  = jedis.rpop(rewardQueue);
			if(null != message  && !message.equals(NIL)) {
				String[] items = message.split(",");
				Values values = new Values(items[0],items[1]);
				msgHolder = new  MessageHolder(values);
				msgHolder.setStream(REWARD_STREAM);
			}
		}
		return msgHolder;
	}

	@Override
	public void handleFailedMessage(Values tuple) {
		// TODO Auto-generated method stub
		
	}

}
