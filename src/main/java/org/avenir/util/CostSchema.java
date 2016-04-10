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

import java.util.List;

/**
 * Manages costs associated with changing attribute values of an entity
 * @author pranab
 *
 */
public class CostSchema {
	private List<CostAttribute> attributes;

	public List<CostAttribute> getAttributes() {
		return attributes;
	}

	public void setAttributes(List<CostAttribute> attributes) {
		this.attributes = attributes;
	}
	
	/**
	 * @param attr
	 * @param valueChange
	 * @return
	 */
	public double findCost(int attr, double valueChange) {
		double cost = 0;
		CostAttribute foundAttr =  findCostAttribute(attr);
		if (null != foundAttr) {
			cost = foundAttr.getNumAttrCost() * valueChange;
		} else {
			throw new IllegalArgumentException("invalid attribute ordinal");
		}
		return cost;
	}
	
	/**
	 * @param attr
	 * @param fromValue
	 * @param toValue
	 * @return
	 */
	public double findCost(int attr, String  fromValue, String  toValue) {
		CostAttribute foundAttr =  findCostAttribute(attr);
		if (null == foundAttr) {
			throw new IllegalArgumentException("invalid attribute ordinal");
		}
		String attrKey = fromValue + "," + toValue;
		Double cost = foundAttr.getCatAttrCost().get(attrKey);
		if (null == cost) {
			//if cost not specified assume 0
			cost = 0.0;
		}
		return cost;
	}
	
	/**
	 * @param attr
	 * @return
	 */
	public CostAttribute findCostAttribute(int attr) {
		CostAttribute foundAttr = null;
		for (CostAttribute costAttr :  attributes) {
			if (costAttr.getOrdinal() == attr) {
				foundAttr = costAttr;
			}
		}
		return foundAttr;
	}
}
