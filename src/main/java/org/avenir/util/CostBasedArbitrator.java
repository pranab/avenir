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


package org.avenir.util;

/**
 * @author pranab
 *
 */
public class CostBasedArbitrator {
	private String posClass;
	private String negClass;
	private int falseNegCost;
	private int falsePosCost;
	
	/**
	 * @param negClass
	 * @param posClass
	 * @param falseNegCost
	 * @param falsePosCost
	 */
	public CostBasedArbitrator(String negClass, String posClass,
			int falseNegCost, int falsePosCost) {
		this.posClass = posClass;
		this.negClass = negClass;
		this.falseNegCost = falseNegCost;
		this.falsePosCost = falsePosCost;
	}
	
	/**
	 * @param posProb
	 * @param negProb
	 * @return
	 */
	public String arbitrate(int posProb, int negProb) {
		int negCost = falseNegCost * posProb + negProb;
		int posCost = falsePosCost * negProb + posProb;
		String predClass =  posCost < negCost  ? posClass : negClass;
		return predClass;
	}
	
	/**
	 * @param posProb
	 * @return
	 */
	public String classify(int posProb) {
		String predClass =  posProb > (falsePosCost * 100) / (falsePosCost + falseNegCost)  ? posClass : negClass;
		return predClass;
	}
}
