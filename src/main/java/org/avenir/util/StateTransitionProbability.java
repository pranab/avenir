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

import org.chombo.util.TabularData;

/**
 * @author pranab
 *
 */
public class StateTransitionProbability extends TabularData {
	private int scale = 100;
	
	public StateTransitionProbability() {
		super();
	}
	
	public StateTransitionProbability(int numRow, int numCol) {
		super(numRow, numCol);
	}

	public StateTransitionProbability(String[] rowLabels, String[] colLabels) {
		super(rowLabels, colLabels);
	}
	
	public void setScale(int scale) {
		this.scale = scale;
	}

	public void normalizeRows() {
		int rowSum = 0;
		for (int r = 0; r < numRow; ++r) {
			rowSum = getRowSum(r);
			for (int c = 0; c < numCol; ++c) {
				table[r][c] = (table[r][c] * scale) / rowSum;
			}
		}
	}
}
