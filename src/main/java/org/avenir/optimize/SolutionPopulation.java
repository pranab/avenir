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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.chombo.util.BasicUtils;

/**
 * @author pranab
 *
 */
public class SolutionPopulation implements Serializable {
	private List<SolutionWithCost> population = new ArrayList<SolutionWithCost>();
	
	/**
	 * 
	 */
	public void initialize() {
		population.clear();
	}
	
	/**
	 * @param solution
	 * @param cost
	 */
	public void add(String solution, double cost) {
		population.add(new SolutionWithCost(solution, cost));
	}
	
	/**
	 * @param solution
	 */
	public void add(SolutionWithCost solution) {
		population.add(solution);
	}
	
	/**
	 * @param index
	 * @return
	 */
	public SolutionWithCost getSolution(int index) {
		return population.get(index);
	}

	/**
	 * 
	 */
	public void sort() {
		Collections.sort(population);
	}
	
	/**
	 * @param topCount
	 */
	public void retainTop(int topCount) {
		int size = population.size();
		if (topCount < size) {
			population.subList(topCount, size).clear();
		} else if (topCount > size) {
			throw new IllegalStateException("top count greater than list size");
		}
	}
	
	/**
	 * @param numSel
	 * @return
	 */
	public List<SolutionWithCost> selectRandom(int numSel) {
		return BasicUtils.selectRandomFromList(population, numSel);
	}
	
	/**
	 * @return
	 */
	public SolutionWithCost getBest() {
		//assuming already sorted
		return population.get(0);
	}
	
	/**
	 * @return
	 */
	public SolutionWithCost findBest() {
		//unsorted
		SolutionWithCost bestMemeber = null;
		for (SolutionWithCost member : population) {
			if (null == bestMemeber || member.getCost() < bestMemeber.getCost()) {
				bestMemeber = member;
			} 
		}
		return bestMemeber;
	}

	/**
	 * @return
	 */
	public SolutionWithCost binaryTournament() {
		List<SolutionWithCost> solnPair = BasicUtils.selectRandomFromList(population, 2);
		SolutionWithCost soln  =  solnPair.get(0).getCost() < solnPair.get(1).getCost() ? 
				solnPair.get(0) : solnPair.get(1);
		return soln;
	}
}
