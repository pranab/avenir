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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.chombo.mr.FeatureField;
import org.chombo.mr.FeatureSchema;
import org.chombo.util.BaseAttribute;
import org.chombo.util.Utility;
import org.hoidla.query.Predicate;

/**
 * @author pranab
 *
 */
public class SplitManager {
	private Map<Integer, String> dataTypes = new HashMap<Integer, String>();
	private List<List<AttributePredicate>> decisionPaths = new ArrayList<List<AttributePredicate>>();
	private FeatureSchema schema;
	private static final String OPERATOR_IN = "in";
	
	/**
	 * @param config
	 * @param statFilePathParam
	 * @param delim
	 * @param schema
	 * @throws IOException
	 */
	public SplitManager(Configuration config, String statFilePathParam,   String delim, FeatureSchema schema) 
			throws IOException {
		super();
		this.schema = schema;
		List<String> lines = Utility.getFileLines(config, statFilePathParam);
		for (String line : lines) {
			//each line is decision path
			List<AttributePredicate> decisionPath = new ArrayList<AttributePredicate>(); 
			decisionPaths.add(decisionPath);
			
			String[] splits = line.split(delim);
			for (String split : splits) {
				String[] splitItems = split.split("\\s+");
				int attr = Integer.parseInt(splitItems[0]);
				String operator = splitItems[1];
				String dataType = dataTypes.get(attr);
				if (null == dataType) {
					dataType = schema.findFieldByOrdinal(attr).getDataType();
					dataTypes.put(attr, dataType);
				}
				
				AttributePredicate pred = null;
				if (dataType.equals(BaseAttribute.DATA_TYPE_INT)) {
					pred = new  IntPredicate(attr, operator, Integer.parseInt(splitItems[2]));
				} else if (dataType.equals(BaseAttribute.DATA_TYPE_DOUBLE)) { 
					pred = new  DoublePredicate(attr, operator, Double.parseDouble(splitItems[2]));
				} else if (dataType.equals(BaseAttribute.DATA_TYPE_CATEGORICAL)) { 
					pred = new  CategoricalPredicate(attr, operator, splitItems[2]);
				}
				decisionPath.add(pred);
			}
		}
	}

	/**
	 * @return
	 */
	public List<Integer> getCurrentAttributes() {
		List<Integer> attributes = new ArrayList<Integer>();
		
		if (!decisionPaths.isEmpty()) {
			List<AttributePredicate> decisionPath = decisionPaths.get(0);
			for (AttributePredicate pred :  decisionPath) {
				attributes.add(pred.attribute);
			}
		}
		return attributes;
	}
	
	/**
	 * @return
	 */
	public List<Integer> getCandidateAttributes() {
		List<Integer> currentAttrs = getCurrentAttributes();
		List<Integer> candidateAttrs = new ArrayList<Integer>();
		
		for (FeatureField field :  schema.getFeatureAttrFields()) {
			if (!currentAttrs.contains(field.getOrdinal())) {
				candidateAttrs.add(field.getOrdinal());
			}
		}
		return candidateAttrs;
	}

	/**
	 * @param attr
	 * @param scanInterVal
	 * @return
	 */
	public List<AttributePredicate> createIntAttrPredicates(int attr) {
		FeatureField field = schema.findFieldByOrdinal(attr);
		double min = field.getMin();
		double max = field.getMax();
		double splitScanInterval = field.getSplitScanInterval();
		
		//number of splits
		int numSplits =(int)( (max - min) / splitScanInterval);
		if (0 == numSplits) {
			numSplits = 1;
			splitScanInterval = (max - min) / 2;
		}
		int[] splitPoints = new int[numSplits];
		
		//split locations
		int splitPoint = (int)(min + splitScanInterval);
		for (int i = 0; i < numSplits; ++i, splitPoint += splitScanInterval) {
			splitPoints[i] = splitPoint;
		}
		
		return createIntAttrPredicates(attr,  splitPoints);
	}
	
	/**
	 * @param attr
	 * @param scanInterVal
	 * @return
	 */
	public List<AttributePredicate> createDoubleAttrPredicates(int attr) {
		FeatureField field = schema.findFieldByOrdinal(attr);
		double min = field.getMin();
		double max = field.getMax();
		double splitScanInterval = field.getSplitScanInterval();
		
		//number of splits
		int numSplits =(int)( (max - min) / splitScanInterval);
		if (0 == numSplits) {
			numSplits = 1;
			splitScanInterval = (max - min) / 2;
		}
		double[] splitPoints = new double[numSplits];
		
		//split locations
		double splitPoint = min + splitScanInterval;
		for (int i = 0; i < numSplits; ++i, splitPoint += splitScanInterval) {
			splitPoints[i] = splitPoint;
		}
		
		return createDoubleAttrPredicates(attr,  splitPoints);
	}

	/**
	 * @param attr
	 * @param splitPoints
	 * @return
	 */
	public List<AttributePredicate> createIntAttrPredicates(int attr, int[] splitPoints) {
		List<AttributePredicate> predicates = new ArrayList<AttributePredicate>();
		for (int i = 0;  i  < splitPoints.length; ++i) {
			if (i == splitPoints.length - 1) {
				AttributePredicate pred = new IntPredicate( attr, Predicate.OPERATOR_LE, splitPoints[i]);
				predicates.add(pred);
				pred = new IntPredicate( attr, Predicate.OPERATOR_GT, splitPoints[i]);
				predicates.add(pred);
			} else if (i == 0) {
				AttributePredicate pred = new IntPredicate( attr, Predicate.OPERATOR_LE, splitPoints[i]);
			} else {
				AttributePredicate pred = new IntPredicate( attr, Predicate.OPERATOR_LE, splitPoints[i], splitPoints[i-1]);
				predicates.add(pred);
			}
		}
		
		return predicates;
	}
	
	/**
	 * @param attr
	 * @param splitPoints
	 * @return
	 */
	public List<AttributePredicate> createDoubleAttrPredicates(int attr,  double[] splitPoints) {
		List<AttributePredicate> predicates = new ArrayList<AttributePredicate>();
		for (int i = 0;  i  < splitPoints.length; ++i) {
			if (i == splitPoints.length - 1) {
				AttributePredicate pred = new DoublePredicate( attr, Predicate.OPERATOR_LE, splitPoints[i]);
				predicates.add(pred);
				pred = new DoublePredicate( attr, Predicate.OPERATOR_GT, splitPoints[i]);
				predicates.add(pred);
			} else if (i == 0) {
				AttributePredicate pred = new DoublePredicate( attr, Predicate.OPERATOR_LE, splitPoints[i]);
			} else {
				AttributePredicate pred = new DoublePredicate( attr, Predicate.OPERATOR_LE, splitPoints[i], splitPoints[i-1]);
				predicates.add(pred);
			}
		}
		
		return predicates;
	}
	
	/**
	 * @param attr
	 * @param groups
	 * @return
	 */
	public List<AttributePredicate> createCategoricalAttrPredicates(int attr,  List<List<String>> groups) {
		List<AttributePredicate> predicates = new ArrayList<AttributePredicate>();
		for (List<String> group :  groups) {
			AttributePredicate pred = new CategoricalPredicate( attr, Predicate.OPERATOR_LE,  group);
			predicates.add(pred);
		}
		return predicates;
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static abstract class AttributePredicate {
		protected int attribute;
		protected String operator;
		
		public AttributePredicate(int attribute, String operator) {
			super();
			this.attribute = attribute;
			this.operator = operator;
		}
		
		/**
		 * @param operand
		 * @return
		 */
		public abstract boolean evaluate(Object operand) ;
		
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static  class IntPredicate extends AttributePredicate {
		private int value; 
		private Integer otherBound;
		
		public IntPredicate(int attribute, String operator, int value) {
			super(attribute, operator);
			this.value = value;
		}
		
		public IntPredicate(int attribute, String operator, int value, Integer otherBound) {
			super(attribute, operator);
			this.value = value;
			this.otherBound = otherBound;
		}
		
		/* (non-Javadoc)
		 * @see org.avenir.tree.SplitManager.AttributePredicate#evaluate(java.lang.Object)
		 */
		public boolean evaluate(Object operandObj) {
			boolean result = false;
			int operand = (Integer)operandObj;
			if (operator.equals(Predicate.OPERATOR_GE)) {
				result = operand >= value;
				if (null != otherBound) {
					result = result && operand < otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_GT)) {
				result = operand > value;
				if (null != otherBound) {
					result = result && operand <= otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_LE)) {
				result = operand < value;
				if (null != otherBound) {
					result = result && operand > otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_LT)) {
				result = operand <= value;
				if (null != otherBound) {
					result = result && operand >= otherBound;
				}
			} else {
				throw new IllegalArgumentException("Illegal int  attribute operator");
			}
					
			return result;
		}
	}

	/**
	 * @author pranab
	 *
	 */
	public static  class DoublePredicate extends AttributePredicate {
		private double value; 
		private Double otherBound;
		
		public DoublePredicate(int attribute, String operator, double value) {
			super(attribute, operator);
			this.value = value;
		}

		public DoublePredicate(int attribute, String operator, double value, Double otherBound) {
			super(attribute, operator);
			this.value = value;
		}

		/* (non-Javadoc)
		 * @see org.avenir.tree.SplitManager.AttributePredicate#evaluate(java.lang.Object)
		 */
		public boolean evaluate(Object operandObj) {
			boolean result = false;
			double operand = (Double)operandObj;
			if (operator.equals(Predicate.OPERATOR_GE)) {
				result = operand >= value;
				if (null != otherBound) {
					result = result && operand < otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_GT)) {
				result = operand > value;
				if (null != otherBound) {
					result = result && operand <= otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_LE)) {
				result = operand < value;
				if (null != otherBound) {
					result = result && operand > otherBound;
				}
			} else if (operator.equals(Predicate.OPERATOR_LT)) {
				result = operand <= value;
				if (null != otherBound) {
					result = result && operand >= otherBound;
				}
			} else {
				throw new IllegalArgumentException("Illegal double attribute operator");
			}
			return result;
		}
	}

	/**
	 * @author pranab
	 *
	 */
	public static  class CategoricalPredicate extends AttributePredicate {
		private List<String> values; 
		
		public CategoricalPredicate(int attribute, String operator, List<String> values) {
			super(attribute, operator);
			this.values = values;
		}

		public CategoricalPredicate(int attribute, String operator, String values) {
			super(attribute, operator);
			String[] valueItems = values.split(",");
			this.values = Arrays.asList(valueItems);
		}
		
		public boolean evaluate(Object operandObj) {
			boolean result = false;
			String operand = (String)operandObj;
			if (operator.equals(OPERATOR_IN)) {
				result = values.contains(operand);
			} else {
				throw new IllegalArgumentException("Illegal categorical attribute operator");
			}
			return result;
		}
	}
}
