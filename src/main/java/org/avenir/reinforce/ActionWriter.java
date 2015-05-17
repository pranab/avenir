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

/**
 * Abstract base class for storing actions selected by RL
 * @author pranab
 *
 */
public interface ActionWriter {
	
	/**
	 * @param stormConf
	 */
	public  void intialize(Map stormConf) ;
	
	/**
	 * @param eventID
	 * @param actions
	 */
	public  void write(String eventID, String[] actions);
	
	/**
	 * @param eventID
	 * @param actions
	 */
	public  void write(String eventID, Action[] actions);
}
