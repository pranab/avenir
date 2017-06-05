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
public class LinearKernel extends Kernel {

	public LinearKernel(int beg, int end) {
		super(beg, end);
	}

	@Override
	public double compute(double[] vec1st, double[] vec2nd) {
		double kernel = 0;
		for (int i = beg; i < end; ++i) {
			kernel += vec1st[i] * vec2nd[i];
		}
		return 0;
	}

}
