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
 * Markov state transition probability matrix
 * @author pranab
 *
 */
public class StateTransitionProbability extends TabularData {
	private int scale = 100;
	private double[][] dTable;
	
	/**
	 * 
	 */
	public StateTransitionProbability() {
		super();
	}
	
	/**
	 * @param numRow
	 * @param numCol
	 */
	public StateTransitionProbability(int numRow, int numCol) {
		super(numRow, numCol);
	}

	/**
	 * @param rowLabels
	 * @param colLabels
	 */
	public StateTransitionProbability(String[] rowLabels, String[] colLabels) {
		super(rowLabels, colLabels);
	}
	
	/**
	 * @param scale
	 */
	public void setScale(int scale) {
		this.scale = scale;
	}

	/**
	 * 
	 */
	public void normalizeRows() {
		//laplace correction
		for (int r = 0; r < numRow; ++r) {
			boolean gotZeroCount = false;
			for (int c = 0; c < numCol && !gotZeroCount; ++c) {
				gotZeroCount = table[r][c] == 0;
			}
			
			if (gotZeroCount) {
				for (int c = 0; c < numCol; ++c) {
					 table[r][c] += 1;
				}			
			}
		}		
		
		//normalize
		int rowSum = 0;
		if (scale == 1) {
			dTable = new double[numRow][numCol];
		}
		for (int r = 0; r < numRow; ++r) {
			rowSum = getRowSum(r);
			for (int c = 0; c < numCol; ++c) {
				if (scale > 1) {
					table[r][c] = (table[r][c] * scale) / rowSum;
				} else {
					dTable[r][c] = ((double)table[r][c]) / rowSum;
				}
			}
		}
	}
	
	/* (non-Javadoc)
	 * @see org.chombo.util.TabularData#serializeRow(int)
	 */
	public String serializeRow(int row) {
		StringBuilder stBld = new StringBuilder();
		for (int c = 0; c < numCol; ++c) {
			if (scale > 1) {
				stBld.append(table[row][c]).append(DELIMETER);
			} else {
				stBld.append(dTable[row][c]).append(DELIMETER);
			}
		}
		
		return stBld.substring(0, stBld.length()-1);
	}
	
	/* (non-Javadoc)
	 * @see org.chombo.util.TabularData#deseralizeRow(java.lang.String, int)
	 */
	public void deseralizeRow(String data, int row) {
		String[] items = data.split(DELIMETER);
		int k = 0;
		for (int c = 0; c < numCol; ++c) {
			if (scale > 1) {
				table[row][c]  = Integer.parseInt(items[k++]);
			} else {
				dTable[row][c]  = Double.parseDouble(items[k++]);
			}
		}
	}
	
	
}
