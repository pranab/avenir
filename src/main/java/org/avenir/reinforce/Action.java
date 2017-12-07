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

import java.io.Serializable;

/**
 * Action object
 * @author pranab
 *
 */
public class Action implements Serializable {
	private String id;
	private long trialCount;
	private double totalReward;
	
	/**
	 * @param id
	 */
	public Action(String id) {
		super();
		this.id = id;
	}
	/**
	 * @return
	 */
	public String getId() {
		return id;
	}
	/**
	 * @param id
	 */
	public void setId(String id) {
		this.id = id;
	}
	
	/**
	 * 
	 */
	public void select() {
		++trialCount;
	}
	
	/**
	 * @param numTrial
	 */
	public void select(int numTrial) {
		trialCount += numTrial;
	}

	/**
	 * @param reward
	 */
	public void reward(double reward) {
		totalReward += reward;
		++trialCount;
	}
	
	/**
	 * @return
	 */
	public long getTrialCount() {
		return trialCount;
	}
	
	public void setTrialCount(long trialCount) {
		this.trialCount = trialCount;
	}
	/**
	 * @return
	 */
	public double getTotalReward() {
		return totalReward;
	}
	
	/**
	 * @param totalReward
	 */
	public void setTotalReward(double totalReward) {
		this.totalReward = totalReward;
	}
	
	/**
	 * @return
	 */
	public double getAverageReward() {
		return trialCount > 0 ? totalReward / trialCount : 0;
	}
}
