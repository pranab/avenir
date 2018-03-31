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
import java.util.List;

/**
 * @author pranab
 *
 */
public class TaxiFleet implements Serializable {
	private List<Taxi> taxis;
	private List<Passenger> passengers;
	private float maxDistance;
	private float maxEarningStdDev;
	
	public List<Taxi> getTaxis() {
		return taxis;
	}
	public void setTaxis(List<Taxi> taxis) {
		this.taxis = taxis;
	}
	public List<Passenger> getPassengers() {
		return passengers;
	}
	public void setPassengers(List<Passenger> passengers) {
		this.passengers = passengers;
	}
	public float getMaxDistance() {
		return maxDistance;
	}
	public void setMaxDistance(float maxDistance) {
		this.maxDistance = maxDistance;
	}
	public float getMaxEarningStdDev() {
		return maxEarningStdDev;
	}
	public void setMaxEarningStdDev(float maxEarningStdDev) {
		this.maxEarningStdDev = maxEarningStdDev;
	}
	
}
