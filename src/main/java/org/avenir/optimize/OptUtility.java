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

import org.chombo.util.Pair;

/**
 * @author pranab
 *
 */
public class OptUtility {
	/**
	 * @param intialSolution
	 * @param cost
	 * @param domain
	 * @param numIteration
	 * @return
	 */
	public static Pair<String, Double> localTrajectorySearch(String intialSolution, BasicSearchDomain domain, 
			int numIteration) {
		double cost = domain.getSolutionCost(intialSolution);
		return localTrajectorySearch(intialSolution, cost, domain, numIteration);
	}
	
	/**
	 * @param intialSolution
	 * @param domain
	 * @param numIteration
	 * @return
	 */
	public static Pair<String, Double> localTrajectorySearch(String intialSolution, double cost, 
			BasicSearchDomain domain, int numIteration) {
		domain.withNeighborhoodReferenceCurrent();
		domain.withCurrentSolution(intialSolution);
		String curSolution = intialSolution;
		String bestSolution = curSolution;
		double curCost = domain.getSolutionCost(intialSolution);
		double bestCost = curCost;
		
		for (int i = 0; i < numIteration; ++i) {
			curSolution = domain.createNeighborhoodSolution();
			curCost = domain.getSolutionCost(curSolution);
			if (curCost < bestCost) {
				//found better solution
				bestSolution = curSolution;
				bestCost = curCost;
				domain.withCurrentSolution(bestSolution);
			}
		}
		return new Pair<String, Double>(bestSolution, bestCost);
	}
}
