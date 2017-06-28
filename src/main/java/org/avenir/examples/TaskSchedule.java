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
import java.text.SimpleDateFormat;
import java.util.List;

/**
 * @author pranab
 *
 */
public class TaskSchedule implements Serializable {
	private List<Location> locations;
	private List<Task> tasks;
	private List<Employee>  employees;
	private double maxPerDiemRate;
	private double[] airFareEstimator;
	private double airTravelDistThreshold;
	private double perMileDriveCost;
	private double maxTravelCost;
	private double costScale;
	private double maxHotelRate;
	private String dateFormat;
	private int minDaysGap;
	private int numComponents;
	private double inavlidSolutionCost;
	
	
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
	public double[] getAirFareEstimator() {
		return airFareEstimator;
	}
	public void setAirFareEstimator(double[] airFareEstimator) {
		this.airFareEstimator = airFareEstimator;
	}
	
	public double getAirTravelDistThreshold() {
		return airTravelDistThreshold;
	}
	public void setAirTravelDistThreshold(double airTravelDistThreshold) {
		this.airTravelDistThreshold = airTravelDistThreshold;
	}
	public double getPerMileDriveCost() {
		return perMileDriveCost;
	}
	public void setPerMileDriveCost(double perMileDriveCost) {
		this.perMileDriveCost = perMileDriveCost;
	}
	public double getMaxPerDiemRate() {
		return maxPerDiemRate;
	}
	public void setMaxPerDiemRate(double maxPerDiemRate) {
		this.maxPerDiemRate = maxPerDiemRate;
	}
	public double getMaxTravelCost() {
		return maxTravelCost;
	}
	public void setMaxTravelCost(double maxTravelCost) {
		this.maxTravelCost = maxTravelCost;
	}
	public double getCostScale() {
		return costScale;
	}
	public void setCostScale(double costScale) {
		this.costScale = costScale;
	}
	public double getMaxHotelRate() {
		return maxHotelRate;
	}
	public void setMaxHotelRate(double maxHotelRate) {
		this.maxHotelRate = maxHotelRate;
	}
	public String getDateFormat() {
		return dateFormat;
	}
	public void setDateFormat(String dateFormat) {
		this.dateFormat = dateFormat;
	}
	public int getMinDaysGap() {
		return minDaysGap;
	}
	public void setMinDaysGap(int minDaysGap) {
		this.minDaysGap = minDaysGap;
	}
	public double getInavlidSolutionCost() {
		return inavlidSolutionCost;
	}
	public void setInavlidSolutionCost(double inavlidSolutionCost) {
		this.inavlidSolutionCost = inavlidSolutionCost;
	}
	/**
	 * @param taskID
	 * @return
	 */
	public Task findTask(String taskID) {
		Task foundTask = null;
		for (Task task : tasks) {
			if (task.getId().equals(taskID)) {
				foundTask = task;
				break;
			}
		}
		return foundTask;
	}
	
	/**
	 * @param employeeID
	 * @return
	 */
	public Employee findEmployee(String employeeID) {
		Employee foundEmployee = null;
		for (Employee employee : employees) {
			if (employee.getId().equals(employeeID)) {
				foundEmployee = employee;
				break;
			}
		}
		return foundEmployee;
	}
	
	/**
	 * @param employeeID
	 * @return
	 */
	public Location findLocation(String locationID) {
		Location foundLocation = null;
		for (Location location : locations) {
			if (location.getId().equals(locationID)) {
				foundLocation = location;
				break;
			}
		}
		return foundLocation;
	}
	
	/**
	 * 
	 */
	public void initialize() {
		numComponents = tasks.size();
	}
	
	/**
	 * @return
	 */
	public int findNumComponents() {
		return numComponents;
	}
}
