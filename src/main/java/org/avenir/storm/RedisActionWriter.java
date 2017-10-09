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

import java.util.Map;

import org.chombo.util.ConfigUtility;
import org.apache.commons.lang3.StringUtils;
import org.avenir.reinforce.Action;
import org.avenir.reinforce.ActionWriter;

import redis.clients.jedis.Jedis;

/**
 * @author pranab
 *
 */
public  class RedisActionWriter implements  ActionWriter {
	private Jedis jedis;
	private String actionQueue;

	@Override
	public void intialize(Map stormConf) {
		//action output queue		
		String redisHost = ConfigUtility.getString(stormConf, "redis.server.host");
		int redisPort = ConfigUtility.getInt(stormConf,"redis.server.port");
		jedis = new Jedis(redisHost, redisPort);
		actionQueue = ConfigUtility.getString(stormConf, "redis.action.queue");
	}

	@Override
	public void write(String eventID, String[] actions) {
		String actionList = actions.length > 1 ? StringUtils.join(actions) : actions[0] ;
		jedis.lpush(actionQueue, eventID + "," + actionList);
	}

	@Override
	public void write(String eventID, Action[] actions) {
		StringBuilder stBld = new StringBuilder();
		for (Action action : actions) {
			stBld.append(action.getId()).append(",");
		}
		jedis.lpush(actionQueue, eventID + "," + stBld.substring(0, stBld.length() -1));
	}
}
