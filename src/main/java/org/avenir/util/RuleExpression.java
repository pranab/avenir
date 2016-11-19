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

import org.chombo.util.AttributeFilter;
import org.chombo.util.BasicUtils;

/**
 * Rule definition consisting of condition and consequence
 * @author pranab
 *
 */
public class RuleExpression extends AttributeFilter {
	private String consequent;
	private static final String CONSQUENT_SEP = ">";
	
	/**
	 * 
	 */
	public RuleExpression() {
		super();
	}

	/**
	 * @param condition
	 */
	public RuleExpression(String condition) {
		super(condition);
	}

	/**
	 * @param rule
	 * @return
	 */
	public static RuleExpression createRule(String rule) {
		String[] items = BasicUtils.splitOnFirstOccurence(rule, CONSQUENT_SEP, true);
		RuleExpression ruleExp = new RuleExpression(items[0].trim());
		ruleExp.setConsequent(items[1].trim());
		return ruleExp;
	}

	/**
	 * @param rule
	 * @return
	 */
	public static String extractConsequent(String rule) {
		String[] items = BasicUtils.splitOnFirstOccurence(rule, CONSQUENT_SEP, true);
		return items[1].trim();
	}
	
	public String getConsequent() {
		return consequent;
	}

	public void setConsequent(String consequent) {
		this.consequent = consequent;
	}
	
}
