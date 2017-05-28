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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chombo.util.BasicUtils;

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
	private double eps;
	private double threshold;
	private List<double[]> data;
	private List<Integer> supVecs = new ArrayList<Integer>();
	private Kernel kernel;
	private Map<Integer, Double> errors = new HashMap<Integer, Double>();
	private Map<int[], Double> kernelValues = new HashMap<int[], Double>();
	public static final String KERNEL_LINER = "linear";
	public static final String KERNEL_POLY = "polynomial";
	public static final String KERNEL_RADIAL = "radial";
	
	/**
	 * @param penaltyFactor
	 * @param classAttrOrd
	 */
	public SequentialMinimalOptimization(double penaltyFactor, int classAttrOrd, double tolerance, 
			double eps, String kernelType) {
		super();
		this.penaltyFactor = penaltyFactor;
		this.classAttrOrd = classAttrOrd;
		this.lagrangianOrd = classAttrOrd + 1;
		this.tolerance = tolerance;
		this.eps = eps;
		
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
		int index2nd = 0;
		
		while(numChanged > 0 || examineAll) {
			numChanged = 0;
			if (examineAll) {
				//will run in between run of the other loop
				index2nd = 0;
				for ( ; index2nd < data.size(); ++index2nd) {
					numChanged += examine(index2nd);
					
				} 
			} else {
				//this loop run run 1 or more times after each run of the other loop
				index2nd = 0;
				for ( ; index2nd < data.size(); ++index2nd) {
					double[] row = data.get(index2nd);
					double lagrange = row[lagrangianOrd];
					if (lagrange > 0 && lagrange < penaltyFactor) {
						numChanged += examine(index2nd);
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
	private int examine(int index2nd) {
		double[] row2nd = data.get(index2nd);
		double target2nd = row2nd[classAttrOrd];
		double alpha2nd = row2nd[lagrangianOrd];
		double err2nd = getError(index2nd);
		
		double r = err2nd * target2nd;
		boolean status = false;
		double[] row1st = null;
		int index1st = 0;
		if (r < -tolerance && alpha2nd < penaltyFactor || r > tolerance && alpha2nd > 0) {
			//choose optimum partner 
			if (!supVecs.isEmpty()) {
				index1st = choosePartner();
				status = step(index1st, index2nd);
			}
			
			//all sv starting at random
			if (!status && !supVecs.isEmpty()) {
				int pos = (int)(Math.random() * supVecs.size());
				
				//from pos to end
				for (int i = pos; i < supVecs.size() && !status; ++i) {
					index1st = supVecs.get(i);
					status = step(index1st, index2nd);
				}
				
				if (!status) {
					//from beginning to pos
					for (int i = 0; i < pos && !status; ++i) {
						index1st = supVecs.get(i);
						status = step(index1st, index2nd);
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
					status = step(index1st, index2nd);
					++index1st;
				}
				
				if (!status) {
					index1st = 0;
					//from pos to end
					for (int i = 0; i < pos && !status; ++i) {
						status = step(index1st, index2nd);
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
	private boolean step(int index1st,  int index2nd) {
		boolean status = false;
		double[] row1st = data.get(index1st);
		double[] row2nd = data.get(index2nd);
		if (index1st != index2nd) {
			double alpha1st = row1st[lagrangianOrd];
			double target1st = row1st[classAttrOrd];
			double error1st = getError(index1st);
			
			double alpha2nd = row2nd[lagrangianOrd];
			double target2nd = row2nd[classAttrOrd];
			double error2nd = getError(index2nd);
			double s = target1st * target2nd;
			
			//limits for alpha
			double high = 0;
			double low = 0;
			if (target1st != target2nd) {
				low = BasicUtils.max(0, alpha2nd - alpha1st);
				high = BasicUtils.min(penaltyFactor, penaltyFactor + alpha2nd - alpha1st);
			} else {
				low = BasicUtils.max(0, alpha1st + alpha2nd - penaltyFactor);
				high = BasicUtils.min(penaltyFactor, alpha1st + alpha2nd);
			}
			
			//second derivative of objective function
			double k11 = kernel.compute(row1st, row1st);
			double k12 = kernel.compute(row1st, row2nd);
			double k22 = kernel.compute(row2nd, row2nd);
			double eta = 2 * k12 - k11 - k22;
			double alpha2ndNew = 0;
			boolean atBound = false;
			if (eta < 0) {
				//max along constraint lne
				alpha2ndNew = alpha2nd - target2nd * (error1st - error2nd) / eta;
				alpha2ndNew = alpha2ndNew < low ? low : (alpha2ndNew  > high ? high : alpha2ndNew);
			} else {
				//max at end of constraint line
				double gama = alpha1st + s * alpha2nd;
				double v1 = evaluate(index1st) + threshold - target1st * alpha1st * getKernelValue(index1st,  index1st) - 
						target2nd * alpha2nd * getKernelValue(index2nd,  index1st);
				double v2 = evaluate(index2nd) + threshold - target1st * alpha1st * getKernelValue(index1st,  index2nd) - 
						target2nd * alpha2nd * getKernelValue(index2nd,  index2nd);

				//objective fun value at boundary
				double alpha2ndLow = low;
				double temp = gama -s * alpha2ndLow;
				double objValLow = temp + alpha2ndLow - 
						0.5 * getKernelValue(index1st,  index1st) * temp * temp - 
						0.5 * getKernelValue(index2nd,  index2nd)  * alpha2ndLow * alpha2ndLow -
						s * getKernelValue(index1st,  index2nd) * temp * alpha2ndLow  -
						target1st * temp  * v1 -
						target2nd  * alpha2ndLow * v2;
				double alpha2ndHigh = high;
				temp = gama -s * alpha2ndHigh;
				double objValHigh = temp + alpha2ndHigh - 
						0.5 * getKernelValue(index1st,  index1st) * temp * temp - 
						0.5 * getKernelValue(index2nd,  index2nd)  * alpha2ndHigh * alpha2ndHigh -
						s * getKernelValue(index1st,  index2nd) * temp * alpha2ndHigh  -
						target1st * temp  * v1 -
						target2nd  * alpha2ndHigh * v2;
				
				//choose by higher value
				if ( objValLow > objValHigh + eps) {
					alpha2ndNew = low;
				} else if (objValLow < objValHigh - eps) {
					alpha2ndNew = high;
				} else {
					alpha2ndNew = alpha2nd;
				}
				atBound = true;
			}
			if (Math.abs(alpha2ndNew - alpha2nd) > eps* (alpha2ndNew + alpha2nd + eps)) {
				double alpha1stNew =  alpha1st + s * (alpha2nd - alpha2ndNew);
				
				//update threshold
				double thresholdNew = 0;
				if (isNotBound(alpha1stNew)) {
					double b1 = error1st + 
							target1st * (alpha1stNew - alpha1st) * getKernelValue(index1st,  index1st) +
							target2nd * (alpha2ndNew - alpha2nd) * getKernelValue(index1st,  index2nd) +
							threshold;
					thresholdNew = b1;
				} else if (isNotBound(alpha2ndNew)) {
					double b2 = error2nd + 
							target1st * (alpha1stNew - alpha1st) * getKernelValue(index1st,  index2nd) +
							target2nd * (alpha2ndNew - alpha2nd) * getKernelValue(index2nd,  index2nd) +
							threshold;
					thresholdNew = b2;
				} else {
					double b1 = error1st + 
							target1st * (alpha1stNew - alpha1st) * getKernelValue(index1st,  index1st) +
							target2nd * (alpha2ndNew - alpha2nd) * getKernelValue(index1st,  index2nd) +
							threshold;
					double b2 = error2nd + 
							target1st * (alpha1stNew - alpha1st) * getKernelValue(index1st,  index2nd) +
							target2nd * (alpha2ndNew - alpha2nd) * getKernelValue(index2nd,  index2nd) +
							threshold;
					thresholdNew = (b1 + b2) / 2;
				}
				
				//update weight vector for linear
				
				//update error cache for non bound lagrangian
				
				//update lagrangian
				data.get(index1st)[lagrangianOrd] = alpha1stNew;
				data.get(index2nd)[lagrangianOrd] = alpha2ndNew;
			}
			
				
		}
		return status;
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
	private double evaluate(int index) {
		double prediction = 0;
		if (!supVecs.isEmpty()) {
			for (int i : supVecs) {
				double[] supVec = data.get(i);
				prediction += supVec[lagrangianOrd] * supVec[classAttrOrd] * getKernelValue(index, i);
			}
			prediction -= threshold;
		}
		return prediction;
	}
	
	/**
	 * @param row
	 * @return
	 */
	private double predict(int indx) {
		double prediction = evaluate(indx) >= 0 ? 1 : 0;
		return prediction;
	}	
	
	/**
	 * @return
	 */
	private int choosePartner() {
		int index1st = 0;
		
		return index1st;
	}
	
	/**
	 * @param indx
	 * @param row
	 * @return
	 */
	private double getError(int index) {
		double[] row = data.get(index);
		Double error = null;
		if (isBound(index)) {
			//bound in cache
			error = errors.get(index);
			if (null == error) {
				error = 0.0;
				errors.put(index, error);
			}			
		} else {
			//always evaluate for non bound
			double target = row[classAttrOrd];
			error = evaluate(index) - target;
		}
		return error;
	}
	
	/**
	 * @param index
	 * @return
	 */
	private boolean isBound(int index) {
		double alpha = data.get(index)[lagrangianOrd];
		return alpha < tolerance || alpha > penaltyFactor - tolerance;
	}
	
	private boolean isNotBound(double alpha) {
		return alpha > 0 && alpha < penaltyFactor;
	}
	
	/**
	 * @param index1st
	 * @param index2nd
	 * @return
	 */
	private double getKernelValue(int index1st,  int index2nd) {
		Double kernelVal = null;
		int[] indexes = new int[2];
		indexes[0] = index1st < index2nd ? index1st : index2nd;
		indexes[1] = index1st >= index2nd ? index1st : index2nd;
		kernelVal = kernelValues.get(indexes);
		if (null == kernel) {
			kernelVal = kernel.compute(data.get(index1st), data.get(index1st));
			kernelValues.put(indexes, kernelVal);
		}
		return kernelVal;
	}
}
