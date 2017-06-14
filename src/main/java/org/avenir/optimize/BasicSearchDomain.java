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

import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import org.chombo.util.BasicUtils;

/**
 * Interface between optimization algorithm and business domain logic
 * @author pranab
 *
 */
public abstract class  BasicSearchDomain implements Serializable {
	protected String currentSolution;
	protected String initialSolution;
	protected boolean refCurrent = true;
	private StepSize stepSize = new StepSize();
	private Map<String, Double> compCosts = new HashMap<String, Double>();
	protected int numComponents;
	
	
	/**
	 * @param configFile
	 * @throws IOException 
	 */
	public abstract void intialize(String configFile) ;
	
	/**
	 * @return
	 */
	public abstract  BasicSearchDomain createClone();
	
	/**
	 * @param solution
	 * @return
	 */
	public abstract String[] getSolutionComponenets(String solution);
	
	/**
	 * @param components
	 * @return
	 */
	public abstract String aggregateSolutionComponenets(String[] components);

	/**
	 * replaces solution component at specified position 
	 * @param comonents
	 * @param index
	 */
	protected abstract void replaceSolutionComponent(String[] comonents, int index);
	
	/**
	 * calculates cost for component
	 * @param comp
	 * @return
	 */
	protected abstract double calculateCost(String comp);
	
	/**
	 * Check whole solution for validity
	 * @param components
	 * @return
	 */
	public abstract boolean isValid(String[] components);
		
	/**
	 * checks partial solution up to specified index for validity
	 * @param componentsex
	 * @param ind
	 * @return
	 */
	public abstract boolean isValid(String[] componentsex, int index);
	
	/**
	 * adds a component at specified position 
	 * @param componenets
	 * @param index
	 */
	protected abstract void addComponent(String[] componenets, int index);
	
	/**
	 * creates initial set of candidates
	 * @return
	 */
	public  String[] createSolutions(int numSolutions) {
		String[] solutions = new String[numSolutions];
		for (int i = 0; i < numSolutions; ++i) {
			solutions[i] = createSolution();
		}
		return solutions;
	}
	
	/**
	 * @param candidate
	 * @return
	 */
	public BasicSearchDomain withCurrentSolution(String solution) {
		currentSolution = solution;
		return this;
	}
	
	/**
	 * @param candidate
	 * @return
	 */
	public BasicSearchDomain withInitialSolution(String solution) {
		initialSolution = solution;
		return this;
	}
	
	/**
	 * creates next candidate based on last candidate
	 * @return
	 */
	public  String createNeighborhoodSolution() {
		String[] components = refCurrent ? getSolutionComponenets(currentSolution) :
			getSolutionComponenets(initialSolution);
		int step = stepSize.getStepSize();
		for (int i = 1; i <= step; ++i) {
			//component to mutate
			int compIndex = BasicUtils.sampleUniform(numComponents);
			replaceSolutionComponent(components, compIndex);
			while (!isValid(components)) {
				compIndex = BasicUtils.sampleUniform(numComponents);
				replaceSolutionComponent(components, compIndex);
			}
		}
		return aggregateSolutionComponenets(components);
	}
	
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
	public  String createSolution() {
		String[] components = new String[numComponents];
		for (int i = 0; i < numComponents; ++i) {
			addComponent(components, i);
			while (!isValid(components,i)) {
				addComponent(components, i);
			}
		}
		return this.aggregateSolutionComponenets(components);
	}
	
	/**
	 * @return
	 */
	public int getStepSize() {
		return stepSize.getStepSize();
	}

	/**
	 * calculates cost for solution
	 * @param candidate
	 * @return
	 */
	public  double getSolutionCost(String solution) {
		double cost = 0;
		String[] components = getSolutionComponenets(solution);
		for (String comp : components) {
			cost += getSolutionComonentCost(comp);
		}
		return cost / getNumComponents();
	}
	
	
	/**
	 * @param comp
	 * @return
	 */
	public double getSolutionComonentCost(String comp) {
		Double cost = compCosts.get(comp);
		if (null == cost) {
			cost = calculateCost(comp);
			compCosts.put(comp, cost);
		}
		return cost;
	}
	
	public  int getNumComponents(){
		return numComponents;
	}
	
	
}
