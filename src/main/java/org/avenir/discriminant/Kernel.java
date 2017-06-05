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

package org.avenir.discriminant;

/**
 * @author pranab
 *
 */
public abstract class Kernel {
	protected int beg;
	protected int end;
	
	/**
	 * @param beg
	 * @param end
	 */
	public Kernel(int beg, int end) {
		super();
		this.beg = beg;
		this.end = end;
	}
	
	public abstract double compute(double[] vec1st, double[] vec2nd);
	
}
