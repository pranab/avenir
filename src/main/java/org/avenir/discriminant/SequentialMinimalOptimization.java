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
	private double tolerance = 0.001;
	private double eps = 0.001;
	private double threshold;
	private List<double[]> data;
	private List<Integer> supVecs = new ArrayList<Integer>();
	private Kernel kernel;
	private Map<Integer, Double> errors = new HashMap<Integer, Double>();
	private Map<int[], Double> kernelValues = new HashMap<int[], Double>();
	private String kernelType;
	private double[] weights;
	public static final String KERNEL_LINER = "linear";
	public static final String KERNEL_POLY = "polynomial";
	public static final String KERNEL_RADIAL = "radial";
	
	/**
	 * @param penaltyFactor
	 * @param classAttrOrd
	 */
	public SequentialMinimalOptimization(double penaltyFactor, int classAttrOrd, double tolerance, 
			double eps, String kernelType, int dimension) {
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
		this.kernelType = kernelType;
		weights = new double[dimension];
		initialize();
	}
	
	
	/**
	 * @param data
	 */
	public void process(List<double[]> data) {
		this.data = data;
		initialize();
		
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
					double alpha = getLagrangian(index2nd);
					if (isNotBound(alpha)) {
						numChanged += examine(index2nd);
					}
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
		int index1st = 0;
		if (r < -tolerance && alpha2nd < penaltyFactor || r > tolerance && alpha2nd > 0) {
			//choose optimum partner 
			if (!supVecs.isEmpty()) {
				index1st = choosePartner(index2nd, err2nd);
				if (index1st >= 0) {
					status = step(index1st, index2nd);
				}
			}
			
			//all sv starting at random
			if (!status && !supVecs.isEmpty()) {
				int pos = (int)(Math.random() * supVecs.size());
				
				//from pos to end
				for (int i = pos, k = 0; k < supVecs.size() && !status; ++k) {
					index1st = supVecs.get(i);
					status = step(index1st, index2nd);
					i = (i + 1) % supVecs.size();
				}
			}
			
			//all starting at random
			if (!status) {
				int pos = (int)(Math.random() * data.size());

				//from pos to end
				int k = 0;
				for (index1st = pos; k < data.size() && !status; ++k) {
					status = step(index1st, index2nd);
					index1st = (index1st + 1) % data.size();
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
			
			if (low == high) {
				return status;
			}
			
			//second derivative of objective function
			double k11 = getKernelValue(index1st, index1st);
			double k12 = getKernelValue(index1st, index2nd);
			double k22 = getKernelValue(index2nd, index2nd);
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
				double objValLow = objFunctionAtBoundary(index1st, index2nd, target1st, target2nd, gama, s, alpha2ndLow, v1, v2);
				double alpha2ndHigh = high;
				double objValHigh = objFunctionAtBoundary(index1st, index2nd, target1st, target2nd, gama, s, alpha2ndHigh, v1, v2);
				
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
				
				//bound check and adjustment
				if (alpha1stNew < 0) {
					alpha2ndNew += s * alpha1stNew;
					alpha1stNew = 0;
				} else if (alpha1stNew > penaltyFactor) {
					alpha2ndNew += s * (alpha1stNew - penaltyFactor);
					alpha1stNew = penaltyFactor;
				}
				
				//update threshold
				double thresholdNew = 0;
				if (isNotBound(alpha1stNew)) {
					thresholdNew = computeThreshold(index1st, error1st, target1st, alpha1stNew, alpha1st,
							index2nd, error2nd, target2nd, alpha2ndNew, alpha2nd, true);
				} else if (isNotBound(alpha2ndNew)) {
					thresholdNew = computeThreshold(index1st, error1st, target1st, alpha1stNew, alpha1st,
							index2nd, error2nd, target2nd, alpha2ndNew, alpha2nd, false);
				} else {
					double b1 = computeThreshold(index1st, error1st, target1st, alpha1stNew, alpha1st,
							index2nd, error2nd, target2nd, alpha2ndNew, alpha2nd, true);
					double b2 = computeThreshold(index1st, error1st, target1st, alpha1stNew, alpha1st,
							index2nd, error2nd, target2nd, alpha2ndNew, alpha2nd, false);
					thresholdNew = (b1 + b2) / 2;
				}
				
				//update weight vector for linear
				if (kernelType.equals(KERNEL_LINER)) {
					updateWeightVector(index1st, target1st, alpha1stNew, 
						alpha1st, index2nd, target2nd, alpha2ndNew, alpha2nd);					
				}
				
				//update lagrangian
				data.get(index1st)[lagrangianOrd] = alpha1stNew;
				data.get(index2nd)[lagrangianOrd] = alpha2ndNew;
				updateNonBoundSupVecs();
				double thresholdDiff = thresholdNew - threshold;
				threshold = thresholdNew;
				
				//update error cache for non bound lagrangian
				updateErrorCache(index1st, target1st, alpha1st, alpha1stNew, 
						index2nd, target2nd, alpha2nd, alpha2ndNew, thresholdDiff);
				
				status = true;
			}
		}
		return status;
	}
	
	/**
	 * 
	 */
	private void initialize() {
		for (double[] row : data) {
			row[lagrangianOrd] = 0;
		}
		errors.clear();
		kernelValues.clear();
		supVecs.clear();
		threshold = 0;
		for (int i = 0; i < weights.length; ++i) {
			weights[i] = 0;
		}		
	}

	/**
	 * @param index1st
	 * @param index2nd
	 * @param target1st
	 * @param target2nd
	 * @param gama
	 * @param s
	 * @param alpha
	 * @param v1
	 * @param v2
	 * @return
	 */
	private double objFunctionAtBoundary(int index1st, int index2nd, double target1st, double target2nd, 
			double gama, double s, double alpha, double v1, double v2) {
		double temp = gama -s * alpha;
		double objVal = temp + alpha - 
				0.5 * getKernelValue(index1st,  index1st) * temp * temp - 
				0.5 * getKernelValue(index2nd,  index2nd)  * alpha * alpha -
				s * getKernelValue(index1st,  index2nd) * temp * alpha  -
				target1st * temp  * v1 -
				target2nd  * alpha * v2;
		return objVal;
	}
	/**
	 * @param index1st
	 * @param error1st
	 * @param target1st
	 * @param alpha1stNew
	 * @param alpha1st
	 * @param index2nd
	 * @param error2nd
	 * @param target2nd
	 * @param alpha2ndNew
	 * @param alpha2nd
	 * @param first
	 * @return
	 */
	private double computeThreshold(int index1st, double error1st, double target1st, double alpha1stNew, double alpha1st,
			int index2nd, double error2nd, double target2nd, double alpha2ndNew, double alpha2nd, boolean first) {
		double b = 0;
		if (first) {
			b = error1st + 
				target1st * (alpha1stNew - alpha1st) * getKernelValue(index1st,  index1st) +
				target2nd * (alpha2ndNew - alpha2nd) * getKernelValue(index1st,  index2nd) +
				threshold;
		} else {
			b = error2nd + 
				target1st * (alpha1stNew - alpha1st) * getKernelValue(index1st,  index2nd) +
				target2nd * (alpha2ndNew - alpha2nd) * getKernelValue(index2nd,  index2nd) +
				threshold;
		}
		
		return b;
	}
	
	/**
	 * @param index1st
	 * @param target1st
	 * @param alpha1st
	 * @param alpha1stNew
	 * @param index2nd
	 * @param target2nd
	 * @param alpha2nd
	 * @param alpha2ndNew
	 * @param thresholdNew
	 */
	private void updateErrorCache(int index1st, double target1st, double alpha1st, double alpha1stNew, 
			int index2nd, double target2nd, double alpha2nd, double alpha2ndNew, double thresholdDiff) {
		double t1 = target1st * (alpha1stNew - alpha1st);
		double t2 = target2nd * (alpha2ndNew - alpha2nd);
		for (int i : supVecs) {
			if (i == index1st || i == index2nd) {
				errors.put(i, 0.0);
			} else {
				double error = errors.get(i);
				double errorNew = error + 
						t1 * getKernelValue(index1st,  i) +
						t2 * getKernelValue(index2nd,  i) -
						thresholdDiff;
				errors.put(i, errorNew);
			}
		}
	}

	/**
	 * @param index1st
	 * @param target1st
	 * @param alpha1stNew
	 * @param alpha1st
	 * @param index2nd
	 * @param target2nd
	 * @param alpha2ndNew
	 * @param alpha2nd
	 */
	private void updateWeightVector(int index1st, double target1st, double alpha1stNew, 
			double alpha1st, int index2nd, double target2nd, double alpha2ndNew,double alpha2nd){
		double t1 = target1st * (alpha1stNew - alpha1st);
		double t2 = target2nd * (alpha2ndNew - alpha2nd);
		double[] dataVec1st = data.get(index1st);
		double[] dataVec2nd = data.get(index2nd);
		for (int i = 0; i < weights.length; ++i) {
			weights[i] += t1 * dataVec1st[i] + t2 * dataVec2nd[i];
		}		
	}

	/**
	 * 
	 */
	private void updateNonBoundSupVecs() {
		supVecs.clear();
		for (int i = 0; i < data.size() ; ++i) {
			double alpha = data.get(i)[lagrangianOrd];
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
		int i = 0;
		for (double[] vec : data) {
			double alpha = vec[lagrangianOrd];
			double target = vec[classAttrOrd];
			if (alpha > 0) {
				prediction += alpha * target * getKernelValue(index, i);
			}
			++i;
		}
		prediction -= threshold;
		return prediction;
	}
	
	/**
	 * @param row
	 * @return
	 */
	public double predict(int indx) {
		double prediction = evaluate(indx) >= 0 ? 1 : 0;
		return prediction;
	}	
	
	/**
	 * @param index2nd
	 * @param err2nd
	 * @return
	 */
	private int choosePartner(int index2nd, double err2nd) {
		int index1st = -1;
		double err1st = 0;
		double maxErrDiff = 0;
		
		//based on max error difference
		for (int i : supVecs) {
			if (i != index2nd) {
				err1st = getError(i);
				double diff = Math.abs(err1st - err2nd);
				if (diff > maxErrDiff) {
					index1st = i;
					maxErrDiff = diff;
				}
			}
		}
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
		if (isNotBound(index)) {
			//if not bound in cache
			error = errors.get(index);
			if (null == error) {
				error = 0.0;
				errors.put(index, error);
			}			
		} else {
			//always evaluate for  bound
			double target = row[classAttrOrd];
			error = evaluate(index) - target;
		}
		return error;
	}
	
	/**
	 * @param index
	 * @return
	 */
	private boolean isNotBound(int index) {
		double alpha = data.get(index)[lagrangianOrd];
		return alpha > 0 && alpha < penaltyFactor;
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
		
		//indexes in sort order for key
		indexes[0] = index1st < index2nd ? index1st : index2nd;
		indexes[1] = index1st >= index2nd ? index1st : index2nd;
		kernelVal = kernelValues.get(indexes);
		if (null == kernel) {
			kernelVal = kernel.compute(data.get(index1st), data.get(index2nd));
			kernelValues.put(indexes, kernelVal);
		}
		return kernelVal;
	}
	
	/**
	 * @param index
	 * @return
	 */
	private double getLagrangian(int index) {
		double[] row = data.get(index);
		return row[lagrangianOrd];
	}

	/**
	 * @return
	 */
	public List<Integer> getNonBoundSupVecIndexes() {
		return supVecs;
	}
	
	/**
	 * @return
	 */
	public List<Integer> getSupVecIndexes() {
		List<Integer> supVecIndexes = new ArrayList<Integer>();
		int i = 0;
		for (double[] vec : data) {
			double alpha = vec[lagrangianOrd];
			if (alpha > 0) {
				supVecIndexes.add(i);
			}
			++i;
		}
		return supVecIndexes;
	}
	
}
