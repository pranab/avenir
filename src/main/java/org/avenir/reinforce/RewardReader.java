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

import java.util.List;
import java.util.Map;

import org.chombo.util.Pair;

/**
 * Reads action rewards from messaging or database systems
 * @author pranab
 *
 */
public interface RewardReader {
	/**
	 * @param stormConf
	 */
	public  void intialize(Map stormConf) ;
	
	/**
	 * @return
	 */
	public List<Pair<String, Integer>> readRewards();

}
