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
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.avenir.model.ProbabilisticPredictiveModel;
import org.avenir.tree.DecisionPathList.DecisionPath;
import org.avenir.tree.DecisionPathList.DecisionPathPredicate;
import org.chombo.util.FeatureField;
import org.chombo.util.FeatureSchema;
import org.chombo.util.Pair;
import org.codehaus.jackson.map.ObjectMapper;

/**
 * @author pranab
 *
 */
public class DecisionTreeModel extends ProbabilisticPredictiveModel {
    private DecisionPathList decPathList;
    private Map<DecisionPathPredicate, Boolean> predicateValues = new HashMap<DecisionPathPredicate, Boolean>();
    private Map<Integer, FeatureField> fields = new HashMap<Integer, FeatureField>();

	public DecisionTreeModel(FeatureSchema schema, InputStream modelStream) throws IOException {
		super(schema);
        if (null != modelStream) {
        	ObjectMapper  mapper = new ObjectMapper();
        	decPathList = mapper.readValue(modelStream, DecisionPathList.class);
        } else {
        	throw new IllegalStateException("null stteam for model");
        }
	}

	/* (non-Javadoc)
	 * @see org.avenir.model.PredictiveModel#predictClassProb(java.lang.String[])
	 */
	@Override
	protected Pair<String, Double> predictClassProb(String[] items) {
		predicateValues.clear();
		DecisionPath decPathMatched = null;
		
		//all decision paths
		for (DecisionPath decPath : decPathList.getDecisionPaths()) {
			//all predicates in a decision path, except the root predicate
			boolean eval = true;
			List<DecisionPathPredicate> predicates = decPath.getPredicates();
			for (int i = 1; i < predicates.size(); ++i) {
				DecisionPathPredicate predicate = predicates.get(i);
				Boolean predEval = predicateValues.get(predicate);
				if (null != predEval) {
					eval = eval && predEval;
				} else {
					int attrOrd = predicate.getAttribute();
					FeatureField field = fields.get(attrOrd);
					if (null == field) {
						field = schema.findFieldByOrdinal(attrOrd);
						fields.put(attrOrd, field);
					}
					predEval = evaluate(predicate, field, items);
					eval = eval && predEval;
					predicateValues.put(predicate, predEval);
				}
				if (!eval) {
					//go to next decision path
					break;
				}
			}
			if (eval) {
				//done
				decPathMatched = decPath;
				break;
			}
		}
		return decPathMatched.doPrediction();
	}
	
	/**
	 * @param predicate
	 * @param field
	 * @param items
	 * @return
	 */
	private boolean evaluate(DecisionPathPredicate predicate, FeatureField field, String[] items ) {
		boolean predEval = false;
		int attrOrd = field.getOrdinal();
		String operator = predicate.getOperator();
		if (field.isInteger()) {
			int operand = Integer.parseInt(items[attrOrd]);
			int operandValue = predicate.getValueInt();
			if (operator.equals(DecisionPathPredicate.OP_LE)) {
				predEval = operand <= operandValue;
			} else if (operator.equals(DecisionPathPredicate.OP_GT)) {
				predEval = operand > operandValue;
				Integer otherBound = predicate.getOtherBoundInt();
				if (null != otherBound) {
					predEval = predEval && operand <= otherBound; 
				}
			} else {
				throw new IllegalStateException("invalid operator type for int attribute");
			}
		} else if (field.isDouble()) {
			double operand = Double.parseDouble(items[attrOrd]);
			double operandValue = predicate.getValueDbl();
			if (operator.equals(DecisionPathPredicate.OP_LE)) {
				predEval = operand <= operandValue;
			} else if (operator.equals(DecisionPathPredicate.OP_GT)) {
				predEval = operand > operandValue;
				Integer otherBound = predicate.getOtherBoundInt();
				if (null != otherBound) {
					predEval = predEval && operand <= otherBound; 
				}
			} else {
				throw new IllegalStateException("invalid operator type for double attribute");
			}
		} else if (field.isCategorical()) {
			String operand = items[attrOrd];
			List<String>operandValue = predicate.getCategoricalValues();
			if (operator.equals(DecisionPathPredicate.OP_IN)) {
				predEval = operandValue.contains(operand);
			} else {
				throw new IllegalStateException("invalid operator type for categorical attribute");
			}
		}
		
		return predEval;
	}

}
