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
 * Contingency matrix for correlation between categorical attributes
 * @author pranab
 *
 */
public class ContingencyMatrix {
	private int[][] table;
	private int numRow;
	private int numCol;
	private static final String DELIMETER = ",";

	public ContingencyMatrix() {
	}
	
	public ContingencyMatrix(int numRow, int numCol) {
		initialize( numRow,  numCol);
	}
	
	public void  initialize(int numRow, int numCol) {
		table = new int[numRow][numCol];
		for (int r = 0; r < numRow; ++r) {
			for (int c = 0; c < numCol; ++c) {
				table[r][c] = 0;
			}
		}
		this.numRow = numRow;
		this.numCol = numCol;
	}

	public void increment(int row, int col) {
		table[row][col] += 1;
	}
	
	public void aggregate(ContingencyMatrix other) {
		for (int r = 0; r < numRow; ++r) {
			for (int c = 0; c < numCol; ++c) {
				table[r][c]  += other.table[r][c];
			}
		}
	}

	public int getRowSum(int row) {
		int sum = 0;
		for (int c = 0; c < numCol; ++c) {
			sum += table[row][c];
		}
		return sum;
	}

	public int getColumnSum(int col) {
		int sum = 0;
		for (int r = 0; r < numRow; ++r) {
			sum += table[r][col];
		}
		return sum;
	}
	
	public int getSum() {
		int sum = 0;
		for (int r = 0; r < numRow; ++r) {
			for (int c = 0; c < numCol; ++c) {
			sum +=table[r][c];
			}
		}
		return sum;
	}
	
	public String serialize() {
		StringBuilder stBld = new StringBuilder();
		for (int r = 0; r < numRow; ++r) {
			for (int c = 0; c < numCol; ++c) {
				stBld.append(table[r][c]).append(DELIMETER);
			}
		}
		
		return stBld.substring(0, stBld.length()-1);
	}
	
	public void deseralize(String data) {
		String[] items = data.split(DELIMETER);
		int k = 0;
		for (int r = 0; r < numRow; ++r) {
			for (int c = 0; c < numCol; ++c) {
				table[r][c]  = Integer.parseInt(items[k++]);
			}
		}
	}
	
	public int cramerIndex(int scale) {
		//row sums
		int[] rowSum = new int[numRow];
		int totalCount = 0;
		for (int i =0; i < numRow; ++ i) {
			rowSum[i] = 0;
			for (int j = 0; j < numCol; ++j) {
				rowSum[i] += table[i][j];
				totalCount +=  table[i][j];;
			}
			rowSum[i] = rowSum[i] == 0 ? 1 : rowSum[i];
		}
		
		//column sums
		int[] colSum = new int[numCol];
		for (int j = 0; j < numCol; ++j) {
			colSum[j] = 0;
			for (int i =0; i < numRow; ++ i) {
				colSum[j] +=  table[i][j];
			}
			colSum[j] = colSum[j] == 0 ? 1 : colSum[j];
		}
		
		//pearson
		int pearson = 0;
		for (int i =0; i < numRow; ++ i) {
			for (int j = 0; j < numCol; ++j) {
					pearson += (totalCount * table[i][j] * table[i][j]) / (rowSum[i] *  colSum[j] );
			}
		}
		pearson -= totalCount;
		
		//cramer
		int smallerDim = numRow < numCol ?  numRow : numCol;
		int cramer = (pearson * scale) / (totalCount * (smallerDim -1));
		
		return cramer;
	}
	
}
