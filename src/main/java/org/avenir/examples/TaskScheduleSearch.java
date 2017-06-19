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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.HashSet;
import java.util.Set;

import org.avenir.optimize.BasicSearchDomain;
import org.chombo.util.BasicUtils;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * @author pranab
 *
 */
public class TaskScheduleSearch extends BasicSearchDomain {
	private TaskSchedule taskSchedule;
	private SimpleDateFormat dateFormatter;
	
	private static final String compDelim = ";";
	private static final String compItemDelim = ":";
	
	public TaskScheduleSearch() {
	}

	@Override
	public void intialize(String configFile, int maxStepSize, boolean debugOn)  {
		try {
			InputStream fs = new FileInputStream(configFile);
			if (null != fs) {
				ObjectMapper mapper = new ObjectMapper();
				taskSchedule = mapper.readValue(fs, TaskSchedule.class);
			}	
		} catch (IOException ex) {
			throw new IllegalStateException("failed to initialize search object " + ex.getMessage());
		}
		taskSchedule.initialize();
		numComponents = taskSchedule.findNumComponents();
		System.out.println("numComponents :" + numComponents);
		dateFormatter = new SimpleDateFormat(taskSchedule.getDateFormat());
		compCosts.clear();
		System.out.println("maxStepSize: " + maxStepSize);
		withMaxStepSize(maxStepSize);
		withConstantStepSize();
		this.debugOn = debugOn;
	}

	@Override
	public BasicSearchDomain createClone() {
		TaskScheduleSearch searchDomain = new TaskScheduleSearch();
		searchDomain.taskSchedule = this.taskSchedule;
		searchDomain.numComponents = this.numComponents;
		searchDomain.dateFormatter = new SimpleDateFormat(taskSchedule.getDateFormat());
		searchDomain.compCosts.clear();
		searchDomain.withMaxStepSize(this.getMaxStepSize());
		searchDomain.withConstantStepSize();
		searchDomain.debugOn = this.debugOn;
		return searchDomain;
	}

	@Override
	public String[] getSolutionComponenets(String solution) {
		return solution.split(compDelim);
	}

	@Override
	public String aggregateSolutionComponenets(String[] components) {
		return BasicUtils.join(components, compDelim);
	}

	@Override
	protected void replaceSolutionComponent(String[] components, int index) {
		String[] items = components[index].split(compItemDelim);
		String task = items[0];
		String employee = items[1];
		System.out.println("existing component: " + components[index]);
		
		//replace employee
		Set<String> excludes = new HashSet<String>();
		excludes.add(employee);
		String replEmployee = selectEmployee(excludes);
		items[1] = replEmployee;
		components[index] = BasicUtils.join(items, compItemDelim);
		System.out.println("replaced component: " + components[index]);
		
		//if replacement employee was assigned already then swap
		for (int i = 0; i < components.length; ++i) {
			items = components[index].split(compItemDelim);
			String thisEmployee = items[1];
			if (i != index && thisEmployee.equals(replEmployee)) {
				items[1] = employee;
				components[i] = BasicUtils.join(items, compItemDelim);
			}
		}
	}

	@Override
	protected double calculateCost(String comp) {
		System.out.println("calculating cost for component: " + comp);
		String[] items = comp.split(compItemDelim);
		String taskID = items[0];
		String employeeID = items[1];
		Task task = taskSchedule.findTask(taskID);
		Employee employee = taskSchedule.findEmployee(employeeID);
		Location taskLocation = taskSchedule.findLocation(task.getLocation());
		double[] taskGeo = taskLocation.getGps();
		Location employeeLocation = taskSchedule.findLocation(employee.getLocation());
		double[] employeeGeo = employeeLocation.getGps();
		
		//travel cost
		double distance  =  BasicUtils.getGeoDistance(taskGeo[0], taskGeo[1], employeeGeo[0], employeeGeo[1]);
		double travelCost = 0;
		if (distance < taskSchedule.getAirTravelDistThreshold()) {
			travelCost = 2 * distance * taskSchedule.getPerMileDriveCost();
		} else {
			double[] airFareEst = taskSchedule.getAirFareEstimator();
			travelCost = airFareEst[0] * distance * distance + airFareEst[1] * distance + airFareEst[2];
		}
		travelCost /= taskSchedule.getMaxTravelCost();
		travelCost *= taskSchedule.getCostScale();
		
		//per diem cost
		long duration = getEndTime(task) - getStartTime(task)  + 4;
		duration /= BasicUtils.MILISEC_PER_DAY;
		//System.out.println("duration days: " + duration);
		
		double perDiemCost = duration * taskLocation.getPerDiemCost();
		perDiemCost /= duration * taskSchedule.getMaxPerDiemRate();
		perDiemCost *= taskSchedule.getCostScale();
		
		//hotel cost
		double hotelCost = duration * taskLocation.getHotelCost();
		hotelCost /= duration * taskSchedule.getMaxHotelRate();
		hotelCost *= taskSchedule.getCostScale();
		
		//skill match cost
		int matchCount = 0;
		for (String empSkill : employee.getSkills()) {
			if (BasicUtils.contains(task.getSkills(), empSkill)) {
				++matchCount;
			}
		}
		int numReqdSkills = task.getSkills().length;
		double skiilMatchCost = ((double)(numReqdSkills - matchCount) *  taskSchedule.getCostScale()) / numReqdSkills;
		
		//System.out.println("travelCost: " + travelCost + " perDiemCost:" + perDiemCost + 
		//		" hotelCost: " + hotelCost + " skiilMatchCost:" + skiilMatchCost);
		double avCost = (travelCost + perDiemCost + hotelCost + skiilMatchCost) / 4.0;
		return avCost;
	}
	
	/**
	 * @param task
	 * @return
	 */
	private long getStartTime(Task task) {
		long time = 0;
		try {
			//System.out.println("task ID: " + task.getId() + " start date: " + task.getStartDate());
			time = dateFormatter.parse(task.getStartDate()).getTime();
		} catch (Exception ex) {
			throw new IllegalStateException("date formatting error" + ex.getMessage());
		}
		return time;
	}

	/**
	 * @param task
	 * @return
	 */
	private long getEndTime(Task task) {
		long time = 0;
		try {
			time = dateFormatter.parse(task.getEndDate()).getTime();
		} catch (Exception ex) {
			throw new IllegalStateException("date formatting error" + ex.getMessage());
		}
		return time;
	}

	@Override
	public boolean isValid(String[] components) {
		return isValid(components, components.length -1);
	}

	/* (non-Javadoc)
	 * @see org.avenir.optimize.BasicSearchDomain#isValid(java.lang.String[], int)
	 */
	@Override
	public boolean isValid(String[] components, int index) {
		boolean valid = true;
		long minGap = taskSchedule.getMinDaysGap() * BasicUtils.MILISEC_PER_DAY;
		for (int i = 0; i <= index; ++i) {
			String[] items = components[i].split(compItemDelim);
			String employeeOneID = items[1];
			Task taskOne = taskSchedule.findTask(items[0]);
			long taskOneStart = getStartTime(taskOne);
			long taskOneEnd = getEndTime(taskOne);
			for (int j = i + 1; j <= index; ++j) {
				items = components[j].split(compItemDelim);
				String employeeTwoID = items[1];
				if (employeeOneID.equals(employeeTwoID)) {
					Task taskTwo = taskSchedule.findTask(items[0]);
					long taskTwoStart = getStartTime(taskTwo);
					long taskTwoEnd = getEndTime(taskTwo);
					System.out.println("taskTwoStart:" + taskTwoStart + " taskOneEnd:" + taskOneEnd +
							" taskOneStart:" + taskOneStart + " taskTwoEnd:" + taskTwoEnd);
					valid = (taskTwoStart - taskOneEnd)  >= minGap || (taskOneStart - taskTwoEnd)  >= minGap;
					if (!valid) {
						break;
					}
				}
			}
			if (!valid) {
				break;
			}
		}
		return valid;
	}

	@Override
	protected void addComponent(String[] componenets, int index) {
		String taskID = taskSchedule.getTasks().get(index).getId();
		String employeeID = BasicUtils.selectRandom(taskSchedule.getEmployees()).getId();
		String comp = taskID + compItemDelim + employeeID;
		componenets[index] = comp;
	}
	
	/**
	 * @param exclude
	 * @return
	 */
	private String selectEmployee(Set<String> excludes) {
		String thisEmployeeID = BasicUtils.selectRandom(taskSchedule.getEmployees()).getId();
		while(excludes.contains(thisEmployeeID)) {
			thisEmployeeID = BasicUtils.selectRandom(taskSchedule.getEmployees()).getId();
		}
		return thisEmployeeID;
	}
}
