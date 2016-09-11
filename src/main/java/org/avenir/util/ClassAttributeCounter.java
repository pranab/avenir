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
 * Counter for class attribute
 * @author pranab
 *
 */
public class ClassAttributeCounter {
	private int posCount;
	private int negCount;
	
	public void initialize() {
		posCount = 0;
		negCount = 0;
	}
	
	public void add(int posCount, int negCount) {
		this.posCount += posCount;
		this.negCount += negCount;
	}
	
	public void update(int posCount, int negCount) {
		this.posCount = posCount;
		this.negCount = negCount;
	}

	public int getPosCount() {
		return posCount;
	}
	public void setPosCount(int posCount) {
		this.posCount = posCount;
	}
	public int getNegCount() {
		return negCount;
	}
	public void setNegCount(int negCount) {
		this.negCount = negCount;
	}
	
	public int getTotalCount() {
		return posCount + negCount;
	}
}
