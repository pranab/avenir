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


package org.avenir.optimize;

import java.io.Serializable;
import org.chombo.util.BasicUtils;

/**
 * @author pranab
 *
 */
public class StepSize implements Serializable {
	private int maxStepSize;
	enum Strategy {
	    Constant,
	    Uniform,
	    Gaussian
	} 
	private Strategy strategy = Strategy.Constant;
	private double mean;
	private double stdDev;
	
	/**
	 * 
	 */
	public StepSize() {
	}

	/**
	 * @param maxStepSize
	 */
	public StepSize(int maxStepSize) {
		this.maxStepSize = maxStepSize;
	}
	
	/**
	 * @param maxStepSize
	 * @return
	 */
	public StepSize withMaxStepSize(int maxStepSize) {
		this.maxStepSize = maxStepSize;
		return this;
	}
	
	/**
	 * @return
	 */
	public StepSize withConstant() {
		strategy = Strategy.Constant;
		return this;
	}

	/**
	 * @return
	 */
	public StepSize withUniform() {
		strategy = Strategy.Uniform;
		return this;
	}

	/**
	 * @param mean
	 * @param stdDev
	 * @return
	 */
	public StepSize withGaussian(double mean, double stdDev) {
		strategy = Strategy.Gaussian;
		this.mean = mean;
		this.stdDev = stdDev;
		return this;
	}

	/**
	 * @return
	 */
	public int getStepSize() {
		int stepSize = 1;
		if (strategy == Strategy.Constant) {
			stepSize = maxStepSize;
		} else if (strategy == Strategy.Constant) {
			stepSize = BasicUtils.sampleUniform(1, maxStepSize);
		}
		return stepSize;
	}
}
