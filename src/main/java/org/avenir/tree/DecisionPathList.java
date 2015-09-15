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


package org.avenir.tree;

import java.util.Arrays;
import java.util.List;


public class DecisionPathList {
	private List<DecisionPath>  decisionPaths;
	
	public List<DecisionPath> getDecisionPaths() {
		return decisionPaths;
	}

	public void setDecisionPaths(List<DecisionPath> decisionPaths) {
		this.decisionPaths = decisionPaths;
	}

	public void addDecisionPath(DecisionPath decPath) {
		decisionPaths.add(decPath);
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class DecisionPath {
		private List<DecisionPathPredicate> predicates;
		private int population;
		private double infoContent;
		private boolean stopped;
		private String status;
		public static final String STATUS_COMPLETED = "completed";
		public static final String STATUS_IN_PROGRESS = "inProgress";
		
		public DecisionPath() {
		}
		
		public DecisionPath(List<DecisionPathPredicate> predicates,
			int population, double infoContent,  boolean stopped) {
			super();
			this.predicates = predicates;
			this.population = population;
			this.infoContent = infoContent;
			this.stopped = stopped;
		}
		
		public List<DecisionPathPredicate> getPredicates() {
			return predicates;
		}
		public void setPredicates(List<DecisionPathList.DecisionPathPredicate> predicates) {
			this.predicates = predicates;
		}
		public int getPopulation() {
			return population;
		}
		public void setPopulation(int population) {
			this.population = population;
		}
		public double getInfoContent() {
			return infoContent;
		}
		public void setInfoContent(double infoContent) {
			this.infoContent = infoContent;
		}
		public boolean isStopped() {
			return stopped;
		}

		public void setStopped(boolean stopped) {
			this.stopped = stopped;
		}

		public String getStatus() {
			return status;
		}
		public void setStatus(String status) {
			this.status = status;
		}
	}
	
	public static class DecisionPathPredicate {
		private int attribute;
		private String operator;
		private int valueInt;
		private double valueDbl;
		private List<String> categoricalValues;
		private Integer otherBoundInt;
		private Double otherBoundDbl;
		
		public static DecisionPathPredicate createIntPredicate(String predicateStr) {
			DecisionPathPredicate predicate = new  DecisionPathPredicate() ;
			String[] items = predicateStr.split("\\s+");
			predicate.setAttribute(Integer.parseInt(items[0]));
			predicate.setOperator(items[1]);
			predicate.setValueInt(Integer.parseInt(items[2]));
			if (items.length == 4) {
				predicate.setOtherBoundInt(Integer.parseInt(items[3]));
			}
			
			return predicate;
		}
		
		public static DecisionPathPredicate createDoublePredicate(String predicateStr) {
			DecisionPathPredicate predicate = new  DecisionPathPredicate() ;
			String[] items = predicateStr.split("\\s+");
			predicate.setAttribute(Integer.parseInt(items[0]));
			predicate.setOperator(items[1]);
			predicate.setValueDbl(Double.parseDouble(items[2]));
			if (items.length == 4) {
				predicate.setOtherBoundDbl(Double.parseDouble(items[3]));
			}
			
			return predicate;
		}

		public static DecisionPathPredicate createCategoricalPredicate(String predicateStr) {
			DecisionPathPredicate predicate = new  DecisionPathPredicate() ;
			String[] items = predicateStr.split("\\s+");
			predicate.setAttribute(Integer.parseInt(items[0]));
			predicate.setOperator(items[1]);
			
			String[] valueArray = items[2].split(":");
			List<String> categoricalValues = Arrays.asList(valueArray);
			predicate.setCategoricalValues(categoricalValues);
			return predicate;
		}

		public int getAttribute() {
			return attribute;
		}
		public void setAttribute(int attribute) {
			this.attribute = attribute;
		}
		public String getOperator() {
			return operator;
		}
		public void setOperator(String operator) {
			this.operator = operator;
		}
		public int getValueInt() {
			return valueInt;
		}
		public void setValueInt(int valueInt) {
			this.valueInt = valueInt;
		}
		public double getValueDbl() {
			return valueDbl;
		}
		public void setValueDbl(double valueDbl) {
			this.valueDbl = valueDbl;
		}
		public List<String> getCategoricalValues() {
			return categoricalValues;
		}
		public void setCategoricalValues(List<String> categoricalValues) {
			this.categoricalValues = categoricalValues;
		}
		public Integer getOtherBoundInt() {
			return otherBoundInt;
		}
		public void setOtherBoundInt(Integer otherBoundInt) {
			this.otherBoundInt = otherBoundInt;
		}
		public Double getOtherBoundDbl() {
			return otherBoundDbl;
		}
		public void setOtherBoundDbl(Double otherBoundDbl) {
			this.otherBoundDbl = otherBoundDbl;
		}
		
	}
}
