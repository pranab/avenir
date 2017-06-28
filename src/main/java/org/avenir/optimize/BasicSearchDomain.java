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
import java.util.HashSet;
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
	protected int maxCrossOverRetryCount;
	protected boolean debugOn;
	
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
	 * @throws IOException 
	 */
	public abstract void intialize(String configFile, int maxStepSize, int mutationRetryCountLimit, boolean debugOn) ;
	
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
		boolean valid = true;
		String[] components = refCurrent ? getSolutionComponenets(currentSolution) :
			getSolutionComponenets(initialSolution);
		int step = stepSize.getStepSize();
		//System.out.println("step: " + step);
		int tryCount = 0;
		for (int i = 1; i <= step; ++i) {
			//component to mutate
			int compIndex = BasicUtils.sampleUniform(numComponents-1);
			String curComp = components[compIndex];
			System.out.println("component to replace: " + compIndex);
			replaceSolutionComponent(components, compIndex);
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
	
	public  int getNumComponents(){
		return numComponents;
	}
	
	/**
	 * @param firstSoln
	 * @param secondSoln
	 * @return
	 */
	public Pair<String, String> crossOver(String firstSoln, String secondSoln) {
		boolean valid = false;
		int maxCrossOverRetryCount = 10;
		Pair<String, String> crossedOver = null;
		String[] firstSolncomp = getSolutionComponenets(firstSoln);
		String[] secondSolncomp = getSolutionComponenets(secondSoln);
		String[] thisFirstSolncomp = new String[numComponents];
		String[] thisSecondSolncomp = new String[numComponents];
		String[] temp = new String[numComponents];
		boolean[] validationStatus = new boolean[2];
		String[] solutions = new String[2];
		Set<Integer> crossOverPoints = new HashSet<Integer>();
		
		//retry loop
		for (int tryCount = 0; !valid && tryCount < maxCrossOverRetryCount; ++tryCount) {
			int crossOverPt = BasicUtils.sampleUniform(1, numComponents-1);
			if (crossOverPoints.contains(crossOverPt)) {
				continue;
			} else {
				crossOverPoints.add(crossOverPt);
			}
			
			BasicUtils.arrayCopy(firstSolncomp, 0, numComponents, thisFirstSolncomp, 0);
			BasicUtils.arrayCopy(secondSolncomp, 0, numComponents, thisSecondSolncomp, 0);
			
			//cross over segments
			BasicUtils.arrayCopy(thisSecondSolncomp, crossOverPt, numComponents, temp, 0);
			BasicUtils.arrayCopy(thisFirstSolncomp, crossOverPt, numComponents, thisSecondSolncomp, crossOverPt);
			BasicUtils.arrayCopy(temp, 0, numComponents - crossOverPt, thisFirstSolncomp, crossOverPt);
			
			validationStatus[0] = isValid(thisFirstSolncomp);
			validationStatus[1] = isValid(thisSecondSolncomp);
			valid = validationStatus[0] && validationStatus[1];
		}
		
		solutions[0] = aggregateSolutionComponenets(thisFirstSolncomp);
		solutions[1] = aggregateSolutionComponenets(thisSecondSolncomp);
		
		//cache invalid solutions
		for (int i = 0; i < 2; ++i) {
			if (!validationStatus[i]) {
				invalidSolutions.add(solutions[i]);
			}
		}	
		crossedOver = new Pair<String, String>(solutions[0], solutions[1]);
		return crossedOver;
		
	}
}
