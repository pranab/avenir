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
 * Contingency matrix for correlation between categorical attributes
 * @author pranab
 *
 */
public class ContingencyMatrix  extends TabularData {
	private int[] rowSum;
	private int[] colSum;
	private int totalCount;
	
	
	public ContingencyMatrix() {
		super();
	}
	
	public ContingencyMatrix(int numRow, int numCol) {
		super(numRow,  numCol);
	}

	
	public void aggregate(ContingencyMatrix other) {
		for (int r = 0; r < numRow; ++r) {
			for (int c = 0; c < numCol; ++c) {
				table[r][c]  += other.table[r][c];
			}
		}
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
	
	private void getAggregates() {
		rowSum = new int[numRow];
		totalCount = 0;
		for (int i =0; i < numRow; ++ i) {
			rowSum[i] = 0;
			for (int j = 0; j < numCol; ++j) {
				rowSum[i] += table[i][j];
				totalCount +=  table[i][j];;
			}
			rowSum[i] = rowSum[i] == 0 ? 1 : rowSum[i];
		}
		
		//column sums
		colSum = new int[numCol];
		for (int j = 0; j < numCol; ++j) {
			colSum[j] = 0;
			for (int i =0; i < numRow; ++ i) {
				colSum[j] +=  table[i][j];
			}
			colSum[j] = colSum[j] == 0 ? 1 : colSum[j];
		}
		
	}
	
	public double  cramerIndex() {
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
		double  pearson = 0;
		for (int i =0; i < numRow; ++ i) {
			for (int j = 0; j < numCol; ++j) {
					pearson += ( (double)table[i][j] * table[i][j]) / ((double)rowSum[i] *  colSum[j] );
			}
		}
		pearson -= 1.0;
		
		//cramer
		int smallerDim = numRow < numCol ?  numRow : numCol;
		double cramer = (pearson) / (smallerDim -1);
		
		return cramer;
	}
	
	private double[] rowSumAsDouble() {
		double[] rowSumDouble = new double[numRow];
		for (int i =0; i < numRow; ++ i) {
			rowSumDouble[i] = (double)rowSum[i] / totalCount;
		}		
		return rowSumDouble;
	}
	
	private double[] colSumAsDouble() {
		double[] colSumDouble = new double[numCol];
		for (int j =0; j< numCol; ++ j) {
			colSumDouble[j] = (double)colSum[j] / totalCount;
		}		
		return colSumDouble;
	}
	
	public double concentrationCoeff() {
		getAggregates() ;
		double[] rowSumDouble = rowSumAsDouble() ;
		double[] colSumDouble = colSumAsDouble();
		
		double  sumOne = 0;
		for (int i =0; i < numRow; ++ i) {
			double elSqSum = 0;
			for (int j = 0; j < numCol; ++j) {
					double elem = (double)table[i][j] / totalCount;
					elSqSum += elem * elem;
			}
			sumOne += elSqSum /  rowSumDouble[i];
		}
		
		double sumTwo = 0;
		for (int j = 0; j < numCol; ++j) {
			sumTwo += colSumDouble[j] * colSumDouble[j] ;
		}
		
		double concCoeff = (sumOne - sumTwo) / (1.0  - sumTwo);
		return concCoeff;
	}
	
	public double uncertaintyCoeff() {
		double uncertainCoeff = 0;
		getAggregates() ;
		double[] rowSumDouble = rowSumAsDouble() ;
		double[] colSumDouble = colSumAsDouble();
		
		double sumOne = 0;
		for (int i =0; i < numRow; ++ i) {
			for (int j = 0; j < numCol; ++j) {
				double elem = (double)table[i][j] / totalCount;
				sumOne += elem  * Math.log10(elem *  colSumDouble[j] / rowSumDouble[i] );
			}
		}

		double sumTwo = 0;
		for (int j = 0; j < numCol; ++j) {
			sumTwo += colSumDouble[j]  * Math.log10(colSumDouble[j]);
		}
		uncertainCoeff = sumOne / sumTwo;
		return uncertainCoeff;
	}
}
