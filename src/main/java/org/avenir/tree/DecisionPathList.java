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
import java.util.Map;

import org.chombo.util.BasicUtils;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Pair;


/**
 * List of decisions paths
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
		if (null == decisionPaths) {
			decisionPaths = new ArrayList<DecisionPath>();
		}
		decisionPaths.add(decPath);
	}
	
	/**
	 * @param totalPopulation
	 * @return
	 */
	public DecisionPathList withTotalPopulation(int totalPopulation) {
		for (DecisionPath decPath : decisionPaths) {
			decPath.withTotalPopulation(totalPopulation);
		}
		return this;
	}
	
	
	/**
	 * @param predcateStrings
	 * @return
	 */
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
	 * @param predcateString
	 * @return
	 */
	public DecisionPath findDecisionPath(String predcateString) {
		DecisionPath foundDecPath = null;
		for (DecisionPath decPath :  decisionPaths) {
			if (decPath.isMatchedByPredicateString(predcateString)) {
				foundDecPath = decPath;
				break;
			}
		}
		
		return foundDecPath;
	}

	/**
	 * @param predicates
	 * @return
	 */
	public static String[] stripSplitId(String[] predicates) {
		 String[] strippedPredicates = new String[predicates.length];
		 for (int i = 0; i < predicates.length; ++i ) {
			 if  (predicates[i].equals(DecisionTreeBuilder.ROOT_PATH)) {
				 strippedPredicates[i] = predicates[i];
			 } else {
				 strippedPredicates[i]  = BasicUtils.splitOnFirstOccurence(predicates[i], DecisionTreeBuilder.SPLIT_DELIM, true)[1];
			 }
		 }
		 return strippedPredicates;
	}
	
	
	/**
	 * Decision path containing a list of predicates
	 * @author pranab
	 *
	 */
	public static class DecisionPath {
		private List<DecisionPathPredicate> predicates;
		private int population;
		private double infoContent;
		private boolean stopped;
		private Map<String, Double> classValPr;
		private String outputClassVal = "";
		private double confidence;
		private double support;
		
		public DecisionPath() {
		}
		
		/**
		 * @param predicates
		 * @param population
		 * @param infoContent
		 * @param stopped
		 */
		public DecisionPath(List<DecisionPathPredicate> predicates,
			int population, double infoContent,  boolean stopped, Map<String, Double> classValPr) {
			super();
			this.predicates = predicates;
			this.population = population;
			this.infoContent = infoContent;
			this.stopped = stopped;
			this.classValPr = classValPr;
			
			findMajorityClass();
		}
		
		/**
		 * @param population
		 * @param infoContent
		 */
		public DecisionPath(int population, double infoContent,  Map<String, Double> classValPr) {
			this.population = population;
			this.infoContent = infoContent;
			this.stopped = false;
			this.classValPr = classValPr;
		}
		
		/**
		 * @param totalPopulation
		 * @return
		 */
		public DecisionPath withTotalPopulation(int totalPopulation) {
			support = (double)population / totalPopulation;
			return this;
		}
		
		/**
		 * 
		 */
		private void findMajorityClass() {
			double majClassProbab = -1.0;
			for (String classVal : classValPr.keySet()) {
				if (classValPr.get(classVal) > majClassProbab) {
					majClassProbab = classValPr.get(classVal);
					outputClassVal = classVal;
				}
			}
			
			double minClassProbab = 1.0 - majClassProbab;
			final double maxConfidence = 100;
			if (minClassProbab > 0) {
				confidence = majClassProbab / minClassProbab;
				confidence = confidence > maxConfidence ?  maxConfidence : confidence;
			} else {
				confidence = maxConfidence;
			}
		}

		/**
		 * @param predcateStrings
		 * @return
		 */
		public boolean isMatchedByPredicates(String[] predcateStrings) {
			boolean matched = true;
			
			if (null == predicates ) {
				//root 
				matched = predcateStrings[0].equals(DecisionTreeBuilder.ROOT_PATH);
			} else {
				int i = 0;
				for (DecisionPathPredicate predicate : predicates) {
					if (!predicate.getPredicateStr().equals(predcateStrings[i++])) {
						matched = false;
						break;
					}
				}
			}
			return matched;
		}
		
		/**
		 * @param predcateString
		 * @return
		 */
		public boolean isMatchedByPredicateString(String predcateString) {
			boolean matched = false;;
			if (null == predicates) {
				matched = predcateString.equals(DecisionTreeBuilder.ROOT_PATH);
			} else {
				matched =  toStringAllPredicate().equals(predcateString);
			}
			return matched;
		}

		/**
		 * @return
		 */
		public List<DecisionPathPredicate> getPredicates() {
			return predicates;
		}
		
		/**
		 * @param predicates
		 */
		public void setPredicates(List<DecisionPathList.DecisionPathPredicate> predicates) {
			this.predicates = predicates;
		}
		
		/**
		 * @return
		 */
		public int getPopulation() {
			return population;
		}
		
		/**
		 * @param population
		 */
		public void setPopulation(int population) {
			this.population = population;
		}
		
		/**
		 * @return
		 */
		public double getInfoContent() {
			return infoContent;
		}
		
		/**
		 * @param infoContent
		 */
		public void setInfoContent(double infoContent) {
			this.infoContent = infoContent;
		}
		
		/**
		 * @return
		 */
		public boolean isStopped() {
			return stopped;
		}

		/**
		 * @param stopped
		 */
		public void setStopped(boolean stopped) {
			this.stopped = stopped;
		}
		
		/**
		 * @return
		 */
		public Map<String, Double> getClassValPr() {
			return classValPr;
		}

		/**
		 * @param classValPr
		 */
		public void setClassValPr(Map<String, Double> classValPr) {
			this.classValPr = classValPr;
		}

		/**
		 * @return
		 */
		public String getOutputClassVal() {
			return outputClassVal;
		}

		/**
		 * @param outputClassVal
		 */
		public void setOutputClassVal(String outputClassVal) {
			this.outputClassVal = outputClassVal;
		}

		/**
		 * @return
		 */
		public double getConfidence() {
			return confidence;
		}

		/**
		 * @param confidence
		 */
		public void setConfidence(double confidence) {
			this.confidence = confidence;
		}

		/**
		 * @return
		 */
		public double getSupport() {
			return support;
		}

		/**
		 * @param support
		 */
		public void setSupport(double support) {
			this.support = support;
		}

		/**
		 * @return
		 */
		public String toStringAllPredicate() {
			List<String> strPredicates = new ArrayList<String>();
			for (DecisionPathPredicate predicate : predicates) {
				strPredicates.add(predicate.toString());
			}
			return BasicUtils.join(strPredicates, SplitManager.getPredDelim());
		}
		
		/**
		 * @return
		 */
		public Pair<String, Double> doPrediction() {
			String predClVal = null;
			double maxProb = 0;
			for (String clVal : classValPr.keySet()) {
				if (classValPr.get(clVal) > maxProb) {
					predClVal = clVal;
					maxProb = classValPr.get(clVal);
				}
			}
			return new Pair<String, Double>(predClVal, maxProb);
		}
	}
	
	/**
	 * Decision path predicate
	 * @author pranab
	 *
	 */
	public static class DecisionPathPredicate {
		private int attribute;
		private String operator;
		private int valueInt;
		private double valueDbl;
		private List<String> categoricalValues;
		private Integer otherBoundInt;
		private Double otherBoundDbl;
		private String predicateStr;
		public static final String OP_LE = "le";
		public static final String OP_GT = "gt";
		public static final String OP_IN = "in";
		

		/**
		 * @param predicateStr
		 * @return
		 */
		public static DecisionPathPredicate createRootPredicate(String predicateStr) {
			DecisionPathPredicate predicate = new  DecisionPathPredicate() ;
			predicate.setPredicateStr(predicateStr);
			return predicate;
		}		

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
	   		if (predicatesStr.equals(DecisionTreeBuilder.ROOT_PATH)) {
	   			predicates.add(DecisionPathPredicate.createRootPredicate(predicatesStr));
	   		} else {
		   		String[] predicateItems = predicatesStr.split(SplitManager.getPredDelim());
		   		for (String predicateItem : predicateItems) {
		   			if(predicateItem.equals(DecisionTreeBuilder.ROOT_PATH)) {
		   				predicates.add(DecisionPathPredicate.createRootPredicate(predicateItem));
		   			} else {
		   				int attr = Integer.parseInt(predicateItem.split("\\s+")[0]);
		   				FeatureField field = schema.findFieldByOrdinal(attr);
		   				DecisionPathList.DecisionPathPredicate  predicate = deserializePredicate(predicateItem, field); 
		   				predicates.add(predicate);
		   			}
		   		}
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
		public int hashCode() {
			return predicateStr.hashCode();
		}
		
		@Override
		public boolean equals(Object obj) {
			DecisionPathPredicate that = (DecisionPathPredicate)obj;
			return predicateStr.equals(that.predicateStr);
		}
		
	}
}
