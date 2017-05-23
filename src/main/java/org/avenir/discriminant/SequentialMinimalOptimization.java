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

import java.util.ArrayList;
import java.util.List;

/**
 * Soft margin optimization for SVM
 * @author pranab 
 *
 */
public class SequentialMinimalOptimization {
	private double penaltyFactor;
	private int classAttrOrd;
	private int lagrangianOrd;
	private double tolerance;
	private double threshold;
	private List<double[]> data;
	private List<Integer> supVecs = new ArrayList<Integer>();
	private Kernel kernel;
	private int index1st;
	private int index2nd;
	public static final String KERNEL_LINER = "linear";
	public static final String KERNEL_POLY = "polynomial";
	public static final String KERNEL_RADIAL = "radial";
	
	/**
	 * @param penaltyFactor
	 * @param classAttrOrd
	 */
	public SequentialMinimalOptimization(double penaltyFactor, int classAttrOrd, double tolerance, String kernelType) {
		super();
		this.penaltyFactor = penaltyFactor;
		this.classAttrOrd = classAttrOrd;
		this.lagrangianOrd = classAttrOrd + 1;
		this.tolerance = tolerance;
		
		if (kernelType.equals(KERNEL_LINER)) {
			kernel = new LinearKernel(0, classAttrOrd);
		} else {
			throw new IllegalStateException("invalid kernel type");
		}
	}
	
	/**
	 * @param data
	 */
	public void process(List<double[]> data, double threshold) {
		this.data = data;
		this.threshold = threshold;
		buildSupVecs();
		
		int numChanged = 0;
		boolean examineAll = true;
		
		while(numChanged > 0 || examineAll) {
			numChanged = 0;
			if (examineAll) {
				//will run in between run of the other loop
				index2nd = 0;
				for (double[] row : data) {
					numChanged += examine(row);
					++index2nd;
				} 
			} else {
				//this loop run run 1 or more times after each run of the other loop
				index2nd = 0;
				for (double[] row : data) {
					double lagrange = row[lagrangianOrd];
					if (lagrange > 0 && lagrange < penaltyFactor) {
						numChanged += examine(row);
					}
					++index2nd;
				}
			} 
			
			if (examineAll) {
				examineAll = false;
			} else if (0 == numChanged){
				examineAll = true;
			}
		}
	}
	
	/**
	 * @param row
	 * @return
	 */
	private int examine(double[] row2nd) {
		double target2nd = row2nd[classAttrOrd];
		double alpha2nd = row2nd[lagrangianOrd];
		double err2nd = predict(row2nd) - target2nd;
		double r = err2nd * target2nd;
		boolean status = false;
		double[] row1st = null;
		if (r < -tolerance && alpha2nd < penaltyFactor || r > tolerance && alpha2nd > 0) {
			//choose optimum partner 
			if (!supVecs.isEmpty()) {
				row1st = choosePartner();
				status = step(row1st, row2nd);
			}
			
			//all sv starting at random
			if (!status && !supVecs.isEmpty()) {
				int pos = (int)(Math.random() * supVecs.size());
				
				//from pos to end
				for (int i = pos; i < supVecs.size() && !status; ++i) {
					index1st = supVecs.get(i);
					row1st = data.get(index1st);
					status = step(row1st, row2nd);
				}
				
				if (!status) {
					//from beginning to pos
					for (int i = 0; i < pos && !status; ++i) {
						index1st = supVecs.get(i);
						row1st = data.get(index1st);
						status = step(row1st, row2nd);
					}
				}
			}
			
			//all starting at random
			if (!status) {
				int pos = (int)(Math.random() * data.size());
				index1st = pos;

				//from pos to end
				for (int i = pos; i < data.size() && !status; ++i) {
					row1st = data.get(i);
					status = step(row1st, row2nd);
					++index1st;
				}
				
				if (!status) {
					index1st = 0;
					//from pos to end
					for (int i = 0; i < pos && !status; ++i) {
						row1st = data.get(i);
						status = step(row1st, row2nd);
						++index1st;
					}
				}
			}
		}
		
		return status ? 1 : 0;
	}
	
	/**
	 * @param row1st
	 * @param row2nd
	 * @return
	 */
	private boolean step(double[] row1st, double[] row2nd) {
		
		return true;
	}

	/**
	 * 
	 */
	private void buildSupVecs() {
		supVecs.clear();
		for (int i = 0; i < data.size(); ++i) {
			double[] row = data.get(i);
			double alpha = row[lagrangianOrd];
			if (alpha > 0 && alpha < penaltyFactor) {
				supVecs.add(i);
			}
		}		
	}
	
	/**
	 * @param row
	 * @return
	 */
	private double predict(double[] row) {
		double prediction = 0;
		if (!supVecs.isEmpty()) {
			for (int i : supVecs) {
				double[] supVec = data.get(i);
				prediction += supVec[lagrangianOrd] * supVec[classAttrOrd] * kernel.compute(row, supVec);
			}
			prediction += threshold;
		}
		return prediction;
	}
	
	/**
	 * @return
	 */
	private double[] choosePartner() {
		double[] row = null;
		
		return row;
	}

}
