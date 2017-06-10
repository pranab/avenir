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

import java.util.List;

/**
 * @author pranab
 *
 */
public class TaskSchedule {
	private List<Location> locations;
	private List<Task> tasks;
	private List<Employee>  employees;
	private int perDiemCost;
	private double[] airFareEstimator;
	
	public List<Location> getLocations() {
		return locations;
	}
	public void setLocations(List<Location> locations) {
		this.locations = locations;
	}
	public List<Task> getTasks() {
		return tasks;
	}
	public void setTasks(List<Task> tasks) {
		this.tasks = tasks;
	}
	public List<Employee> getEmployees() {
		return employees;
	}
	public void setEmployees(List<Employee> employees) {
		this.employees = employees;
	}
	public int getPerDiemCost() {
		return perDiemCost;
	}
	public void setPerDiemCost(int perDiemCost) {
		this.perDiemCost = perDiemCost;
	}
	public double[] getAirFareEstimator() {
		return airFareEstimator;
	}
	public void setAirFareEstimator(double[] airFareEstimator) {
		this.airFareEstimator = airFareEstimator;
	}
	
}
