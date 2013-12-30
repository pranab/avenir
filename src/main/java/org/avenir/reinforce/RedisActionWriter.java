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

import org.chombo.util.Utility;

import redis.clients.jedis.Jedis;

/**
 * @author pranab
 *
 */
public  class RedisActionWriter extends  ActionWriter {
	private Jedis jedis;
	private String actionQueue;

	@Override
	public void intialize(Map stormConf) {
		//action output queue		
		String redisHost = stormConf.get("redis.server.host").toString();
		int redisPort = new Integer(stormConf.get("redis.server.port").toString());
		jedis = new Jedis(redisHost, redisPort);
		actionQueue =  stormConf.get("redis.action.queue").toString();
	}

	@Override
	public void write(String eventID, String[] actions) {
		String actionList = actions.length > 1 ? Utility.join(actions) : actions[0] ;
		jedis.lpush(actionQueue, eventID + "," + actionList);
	}

}
