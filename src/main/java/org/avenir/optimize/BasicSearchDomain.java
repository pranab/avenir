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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.chombo.util.BasicUtils;
import org.chombo.util.Pair;

/**
 * Interface between optimization algorithm and business domain logic
 * @author pranab
 *
 */
public abstract class  BasicSearchDomain implements Serializable {
	protected String currentSolution;
	protected String initialSolution;
	protected boolean refCurrent;
	protected StepSize stepSize;
	protected Map<String, Double> compCosts;
	protected int numComponents;
	protected int mutationRetryCountLimit;
	protected Set<String> invalidSolutions;
	protected boolean debugOn;
	protected static String compDelim = ";";
	protected static String compItemDelim = ":";
	
	/**
	 * 
	 */
	public BasicSearchDomain() {
		refCurrent = true;
		stepSize = new StepSize();
		compCosts = new HashMap<String, Double>();
		invalidSolutions = new HashSet<String>();
	}
	
	public void reset() {
		compCosts.clear();
	}
	
	/**
	 * @param configFile
	 * @param maxStepSize
	 * @param mutationRetryCountLimit
	 * @param debugOn
	 */
	public abstract void initTrajectoryStrategy(String configFile, int maxStepSize, int mutationRetryCountLimit, 
			boolean debugOn) ;
	
	/**
	 * @return
	 */
	public abstract  BasicSearchDomain createTrajectoryStrategyClone();
	

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
	protected abstract void addSolutionComponent(String[] componenets, int index);
	
	/**
	 * @return
	 */
	protected abstract double getInvalidSolutionCost();
	
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
		if (debugOn) {
			System.out.println("currentSolution before creating new:" + currentSolution);
		}
		String solution =  refCurrent ? currentSolution :initialSolution;
		return mutateSolution(solution);
	}
	
	/**
	 * the extent of neighborhood to use  for neighborhood based candidate
	 * generation
	 * @param size
	 */
	public BasicSearchDomain withMaxStepSize(int maxStepSize) {
		stepSize.withMaxStepSize(maxStepSize);
		return this;
	}
	
	/**
	 * @return
	 */
	public int getMaxStepSize() {
		return stepSize.getStepSize();
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
	 * @return
	 */
	public BasicSearchDomain withNeighborhoodReferenceCurrent() {
		this.refCurrent = true;
		return this;
	}
	
	/**
	 * @return
	 */
	public BasicSearchDomain withNeighborhoodReferenceInitial() {
		this.refCurrent = false;
		return this;
	}

	/**
	 * @param mutationRetryCountLimit
	 * @return
	 */
	public BasicSearchDomain withMutationRetryCountLimit(int mutationRetryCountLimit) {
		this.mutationRetryCountLimit = mutationRetryCountLimit;
		return this;
	}
	
	/**
	 * creates random candidate
	 * @return
	 */
	public  String createSolution() {
		prepareCreateSolution();
		String[] components = new String[numComponents];
		for (int i = 0; i < numComponents; ++i) {
			addSolutionComponent(components, i);
			while (!isValid(components,i)) {
				addSolutionComponent(components, i);
			}
		}
		return this.aggregateSolutionComponenets(components);
	}
	
	public void prepareCreateSolution() {
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
	public double getSolutionCost(String solution) {
		double cost = 0;
		if (invalidSolutions.contains(solution)) {
			cost = getInvalidSolutionCost();
			if (debugOn) {
				System.out.println("returning cost for invalid solution");
			}
		} else {
			String[] components = getSolutionComponenets(solution);
			for (String comp : components) {
				double compCost = getSolutionComonentCost(comp);
				if (debugOn) {
					System.out.println("component: " + comp + " compCost:" + compCost);
				}
				cost += compCost;
			}
			if (debugOn) {
				System.out.println("total cost: " + cost + " num components: " + numComponents);
			}
			cost /= numComponents;
		}
		
		return cost;
	}
	
	
	/**
	 * @param comp
	 * @return
	 */
	public double getSolutionComonentCost(String comp) {
		Double cost = compCosts.get(comp);
		if (null == cost) {
			if (debugOn) {
				System.out.println("missing component cost in cache: " + comp);
			}
			cost = calculateCost(comp);
			compCosts.put(comp, cost);
		} else {
			if (debugOn) {
				//System.out.println("found component cost in cache: " + comp + " cost: " + cost);
			}
		}
		
		return cost;
	}
	
	/**
	 * @return
	 */
	public  int getNumComponents(){
		return numComponents;
	}
	
	/**
	 * 
	 */
	public void prepareMutateSolution() {
	}
	
	/**
	 * @param solution
	 * @param numSolutions
	 * @return
	 */
	public List<String> mutateSolution(String solution, int numSolutions) {
		List<String> solutions = new ArrayList<String>();
		for (int i = 0; i < numSolutions; ++i) {
			solutions.add(mutateSolution(solution));
		}
		return solutions;
	}
		
	/**
	 * @param solution
	 * @return
	 */
	
	public String mutateSolution(String solution) {
		String[] components = getSolutionComponenets(solution);
		boolean valid = false;
		
		int step = stepSize.getStepSize();
		if (step > numComponents) {
			throw new IllegalStateException("mutation step size should not be greater than number of solution components");
		}
		
		//System.out.println("step: " + step);
		int tryCount = 0;
		Set<Integer> selectedComps = new HashSet<Integer>();
		for (int i = 1; i <= step; ++i) {
			//component to mutate
			int compIndex = selectComponentToMutate(selectedComps);
			String curComp = components[compIndex];
			System.out.println("component to replace: " + compIndex);
			replaceSolutionComponent(components, compIndex);
			
			//check validity
			valid = isValid(components);
			tryCount = 0;
			while (!valid && tryCount < mutationRetryCountLimit) {
				components[compIndex] = curComp;
				compIndex = BasicUtils.sampleUniform(numComponents-1);
				System.out.println("found invalid choosing another component to replace: " + compIndex +
						" tryCount: " + tryCount);
				curComp = components[compIndex];
				replaceSolutionComponent(components, compIndex);
				valid = isValid(components);
				++tryCount;
			}
		}
		String newSoln =  aggregateSolutionComponenets(components);
		if (!valid && tryCount == mutationRetryCountLimit) {
			if (debugOn) {
				System.out.println("max retry limit reached creating new solution");
			}
			invalidSolutions.add(newSoln);
		}
		if (debugOn) {
			System.out.println("created newSoln: " + newSoln);
		}
		return newSoln;
	}
	
	/**
	 * @param selectedComps
	 * @return
	 */
	private int selectComponentToMutate(Set<Integer> selectedComps) {
		int comp = BasicUtils.sampleUniform(numComponents-1);
		while(selectedComps.contains(comp)) {
			comp = BasicUtils.sampleUniform(numComponents-1);
		}
		selectedComps.add(comp);
		return comp;
	}
	
	/**
	 * @param solution
	 * @return
	 */
	public String[] getSolutionComponenets(String solution) {
		return solution.split(compDelim);
	}

	/**
	 * @param components
	 * @return
	 */
	public String aggregateSolutionComponenets(String[] components) {
		return BasicUtils.join(components, compDelim);
	}
	
	/**
	 * @param component
	 * @return
	 */
	public String[] getSolutionComponentItems(String component) {
		return component.split(compItemDelim);
	}
	
	
	/**
	 * @param items
	 * @return
	 */
	public String aggregateSolutionComponenetItems(String[] items) {
		return BasicUtils.join(items, compItemDelim);
	}
}
