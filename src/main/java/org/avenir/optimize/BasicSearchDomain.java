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

import org.avenir.optimize.StepSize.Strategy;

/**
 * Interface between optimization algorithm and business domain logic
 * @author pranab
 *
 */
public abstract class  BasicSearchDomain implements Serializable {
	protected String currentCandidate;
	protected String initialCandidate;
	protected boolean refCurrent = true;
	private StepSize stepSize = new StepSize();
	
	
	/**
	 * @param configFile
	 */
	public abstract void intialize(String configFile);
	
	/**
	 * @return
	 */
	public abstract  BasicSearchDomain createClone();
	
	/**
	 * creates initial set of candidates
	 * @return
	 */
	public abstract String[] createCandidates(int numCandidates);
	
	/**
	 * @param candidate
	 * @return
	 */
	public BasicSearchDomain withCurrentCandidate(String candidate) {
		currentCandidate = candidate;
		return this;
	}
	
	/**
	 * @param candidate
	 * @return
	 */
	public BasicSearchDomain withInitialCandidate(String candidate) {
		initialCandidate = candidate;
		return this;
	}
	
	/**
	 * creates next candidate based on last candidate
	 * @return
	 */
	public abstract String createNeighborhoodCandidate();
	
	/**
	 * the extent of neighborhood to use  for neighborhood based candidate
	 * generation
	 * @param size
	 */
	public BasicSearchDomain withMaxSize(int maxStepSize) {
		stepSize.withMaxStepSize(maxStepSize);
		return this;
	}
	

	/**
	 * @return
	 */
	public BasicSearchDomain withConstantStepSize() {
		stepSize.withConstant();
		return this;
	}

	/**
	 * @return
	 */
	public BasicSearchDomain withUniformStepSize() {
		stepSize.withUniform();
		return this;
	}

	/**
	 * @param mean
	 * @param stdDev
	 * @return
	 */
	public BasicSearchDomain withGaussianStepSize(double mean, double stdDev) {
		stepSize.withGaussian(mean, stdDev);
		return this;
	}
	
	
	/**
	 * sets the reference for neighborhood based candidate generation either
	 * current or initial
	 * @param current
	 */
	public BasicSearchDomain withNeighborhoodReference(boolean refCurrent)  {
		this.refCurrent = refCurrent;
		return this;
	}
	
	/**
	 * creates random candidate
	 * @return
	 */
	public abstract String createCandidate();
	
	/**
	 * @return
	 */
	public int getStepSize() {
		return stepSize.getStepSize();
	}

	/**
	 * calculates cost for candidate
	 * @param candidate
	 * @return
	 */
	public abstract double getCandidateCost(String candidate);
}
