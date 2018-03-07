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

package org.avenir.examples;

import java.io.Serializable;

/**
 * @author pranab
 *
 */
public class Passenger implements Serializable {
	private String id;
	private float[] curLocation;
	private float[] destLocation;
	private boolean frequentUser;
	
	public String getId() {
		return id;
	}
	public void setId(String id) {
		this.id = id;
	}
	public float[] getCurLocation() {
		return curLocation;
	}
	public void setCurLocation(float[] curLocation) {
		this.curLocation = curLocation;
	}
	public float[] getDestLocation() {
		return destLocation;
	}
	public void setDestLocation(float[] destLocation) {
		this.destLocation = destLocation;
	}
	public boolean isFrequentUser() {
		return frequentUser;
	}
	public void setFrequentUser(boolean frequentUser) {
		this.frequentUser = frequentUser;
	}
	
}
