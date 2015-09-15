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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.chombo.mr.FeatureField;
import org.chombo.mr.FeatureSchema;


/**
 * @author pranab
 *
 */
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
	
	public DecisionPath findDecisionPath(String[] predcateStrings) {
		DecisionPath foundDecPath = null;
		for (DecisionPath decPath :  decisionPaths) {
			if (decPath.isMatchedByPredicates(predcateStrings)) {
				foundDecPath = decPath;
				break;
			}
		}
		
		return foundDecPath;
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
		
		public boolean isMatchedByPredicates(String[] predcateStrings) {
			boolean matched = true;;
			int i = 0;
			for (DecisionPathPredicate predicate : predicates) {
				if (!predicate.getPredicateStr().equals(predcateStrings[i++])) {
					matched = false;
					break;
				}
			}
			
			return matched;
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
	}
	
	public static class DecisionPathPredicate {
		private int attribute;
		private String operator;
		private int valueInt;
		private double valueDbl;
		private List<String> categoricalValues;
		private Integer otherBoundInt;
		private Double otherBoundDbl;
		private String predicateStr;
		
		/**
		 * @param predicateStr
		 * @return
		 */
		public static DecisionPathPredicate createIntPredicate(String predicateStr) {
			DecisionPathPredicate predicate = new  DecisionPathPredicate() ;
			String[] items = predicateStr.split("\\s+");
			predicate.setAttribute(Integer.parseInt(items[0]));
			predicate.setOperator(items[1]);
			predicate.setValueInt(Integer.parseInt(items[2]));
			if (items.length == 4) {
				predicate.setOtherBoundInt(Integer.parseInt(items[3]));
			}
			predicate.setPredicateStr(predicateStr);
			return predicate;
		}
		
		/**
		 * @param predicateStr
		 * @return
		 */
		public static DecisionPathPredicate createDoublePredicate(String predicateStr) {
			DecisionPathPredicate predicate = new  DecisionPathPredicate() ;
			String[] items = predicateStr.split("\\s+");
			predicate.setAttribute(Integer.parseInt(items[0]));
			predicate.setOperator(items[1]);
			predicate.setValueDbl(Double.parseDouble(items[2]));
			if (items.length == 4) {
				predicate.setOtherBoundDbl(Double.parseDouble(items[3]));
			}
			predicate.setPredicateStr(predicateStr);
			return predicate;
		}

		/**
		 * @param predicateStr
		 * @return
		 */
		public static DecisionPathPredicate createCategoricalPredicate(String predicateStr) {
			DecisionPathPredicate predicate = new  DecisionPathPredicate() ;
			String[] items = predicateStr.split("\\s+");
			predicate.setAttribute(Integer.parseInt(items[0]));
			predicate.setOperator(items[1]);
			
			String[] valueArray = items[2].split(":");
			List<String> categoricalValues = Arrays.asList(valueArray);
			predicate.setCategoricalValues(categoricalValues);
			predicate.setPredicateStr(predicateStr);
			return predicate;
		}

	   	/**
	   	 * @param predicatesStr
	   	 * @param schema
	   	 * @return
	   	 */
	   	public static List< DecisionPathList.DecisionPathPredicate> createPredicates(String predicatesStr, FeatureSchema schema) {
	   		List< DecisionPathList.DecisionPathPredicate> predicates = new ArrayList< DecisionPathList.DecisionPathPredicate>();
	   		String[] predicateItems = predicatesStr.split(";");
	   		for (String predicateItem : predicateItems) {
	   			int attr = Integer.parseInt(predicateItem.split("\\s+")[0]);
	   			FeatureField field = schema.findFieldByOrdinal(attr);
	   			DecisionPathList.DecisionPathPredicate  predicate = deserializePredicate(predicateItem, field); 
	   			predicates.add(predicate);
	   		}
	   		return predicates;
	   	}
		
	   	/**
	   	 * @param predicateStr
	   	 * @param field
	   	 * @return
	   	 */
	   	public static DecisionPathList.DecisionPathPredicate  deserializePredicate(String predicateStr, FeatureField field) {
				DecisionPathList.DecisionPathPredicate predicate = null;
				if (field.isInteger()) {
					predicate = DecisionPathList.DecisionPathPredicate.createIntPredicate(predicateStr);
				} else if (field.isDouble()) {
					predicate = DecisionPathList.DecisionPathPredicate.createDoublePredicate(predicateStr);
				} else if (field.isCategorical()) {
					predicate = DecisionPathList.DecisionPathPredicate.createCategoricalPredicate(predicateStr);
				} else {
					throw new IllegalArgumentException("invalid data type for predicates");
				}
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

		public String getPredicateStr() {
			return predicateStr;
		}

		public void setPredicateStr(String predicateStr) {
			this.predicateStr = predicateStr;
		}

		@Override
		public boolean equals(Object obj) {
			DecisionPathPredicate that = (DecisionPathPredicate)obj;
			return predicateStr.equals(that.predicateStr);
		}
		
	}
}
