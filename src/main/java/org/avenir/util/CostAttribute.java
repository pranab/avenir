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

import java.util.Map;

import org.chombo.util.BaseAttribute;

/**
 * Manages cost associated with changing attribute value
 * @author pranab
 *
 */
public class CostAttribute  extends BaseAttribute {
	private double numAttrCost;
	private Map<String , Double> catAttrCost;
	
	public double getNumAttrCost() {
		return numAttrCost;
	}
	public void setNumAttrCost(double numAttrCost) {
		this.numAttrCost = numAttrCost;
	}
	public Map<String, Double> getCatAttrCost() {
		return catAttrCost;
	}
	public void setCatAttrCost(Map<String, Double> catAttrCost) {
		this.catAttrCost = catAttrCost;
	}
	
}
