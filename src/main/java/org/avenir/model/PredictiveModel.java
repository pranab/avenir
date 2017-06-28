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


package org.avenir.model;

import org.chombo.util.FeatureSchema;
import org.chombo.util.Pair;

/**
 * abstract model predictor
 * @author pranab
 *
 */
public abstract class PredictiveModel {
	protected FeatureSchema schema;
	protected boolean errorCountingEnabled;
	protected int classAttributeOrd;
	protected String posClass; 
	protected String negClass;
	protected String predClass;
	protected boolean costBasedPredictionEnabled;
	protected double falsePosCost;
	protected double falseNegCost;
	protected String[] items;
	protected Pair<String, Double> predClassProb;
	private int totalCount;
	private int errorCount;
	private int falsePosErrorCount;
	private int falseNegErrorCount;
	

	/**
	 * 
	 */
	public PredictiveModel() {
	}
	
	/**
	 * @param schema
	 */
	public PredictiveModel(FeatureSchema schema) {
		this.schema = schema;
	}
	
	/**
	 * @param classAttributeOrd
	 * @param posClass
	 * @param negClass
	 */
	public PredictiveModel enableErrorCounting(int classAttributeOrd, String posClass, String negClass) {
		errorCountingEnabled = true;
		this.classAttributeOrd = classAttributeOrd;
		withClassValues(posClass, negClass);
		return this;
	}
	
	/**
	 * @param falsePosCost
	 * @param falseNegCost
	 */
	public PredictiveModel enableCostBasedPrediction(String posClass, String negClass, 
			double falsePosCost, double falseNegCost) {
		costBasedPredictionEnabled = true;
		withClassValues(posClass, negClass);
		this.falsePosCost = falsePosCost;
		this.falseNegCost = falseNegCost;
		return this;
	}
	
	/**
	 * @param posClass
	 * @param negClass
	 */
	public void withClassValues(String posClass, String negClass) {
		if (null == this.posClass) {
			this.posClass = posClass;
			this.negClass = negClass;
		}
	}
	
	/**
	 * 
	 */
	protected void countError() {
		++totalCount;
		String actualClass = items[classAttributeOrd];
		if (!actualClass.equals(predClass)) {
			if (predClass.equals(posClass)) {
				++falsePosErrorCount;
			} else {
				++falseNegErrorCount;
			}
			++errorCount;
		}
	}

	/**
	 * @param items
	 * @return
	 */
	public abstract String predict(String[] items);
	
	/**
	 * @param items
	 * @return
	 */
	protected abstract Pair<String, Double>  predictClassProb(String[] items);
	
	/**
	 * @return
	 */
	public boolean isErrorCountingEnabled() {
		return errorCountingEnabled;
	}

	/**
	 * @return
	 */
	public double getError() {
		double error = 0;
		if (errorCountingEnabled) {
			error = ((double)errorCount) / totalCount;
		}
		else {
			throw new IllegalStateException("error counting is not enabled");
		}
		return error;
	}
	
	/**
	 * @return
	 */
	public double getFalsePosError() {
		double error = 0;
		if (errorCountingEnabled) {
			error = ((double)falsePosErrorCount) / totalCount;
		}
		else {
			throw new IllegalStateException("error counting is not enabled");
		}
		return error;
	}

	/**
	 * @return
	 */
	public double getFalseNegError() {
		double error = 0;
		if (errorCountingEnabled) {
			error = ((double)falseNegErrorCount) / totalCount;
		}
		else {
			throw new IllegalStateException("error counting is not enabled");
		}
		return error;
	}
}
