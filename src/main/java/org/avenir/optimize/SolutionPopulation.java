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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.chombo.util.BasicUtils;

/**
 * @author pranab
 *
 */
public class SolutionPopulation {
	private List<SolutionWithCost> population = new ArrayList<SolutionWithCost>();
	
	/**
	 * @param solution
	 * @param cost
	 */
	public void add(String solution, double cost) {
		population.add(new SolutionWithCost(solution, cost));
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
		if (topCount < population.size()) {
			population.subList(0, topCount).clear();
		} else if (topCount > population.size()) {
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
		return population.get(0);
	}
}
