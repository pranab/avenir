/*
 * avenir-spark: Predictive analytic based on Spark
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

public class Location implements Serializable {
	private String id;
	private String name;
	private double[] gps;
	private int hotelCost;
	private int perDiemCost;
	
	public String getId() {
		return id;
	}
	public void setId(String id) {
		this.id = id;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public double[] getGps() {
		return gps;
	}
	public void setGps(double[] gps) {
		this.gps = gps;
	}
	public int getHotelCost() {
		return hotelCost;
	}
	public void setHotelCost(int hotelCost) {
		this.hotelCost = hotelCost;
	}
	public int getPerDiemCost() {
		return perDiemCost;
	}
	public void setPerDiemCost(int perDiemCost) {
		this.perDiemCost = perDiemCost;
	}

}
