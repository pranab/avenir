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

import java.util.HashSet;
import java.util.Set;

import org.chombo.util.BasicUtils;
import org.chombo.util.Pair;

/**
 * @author pranab
 *
 */
public abstract class PopulationSearchDomain extends BasicSearchDomain{
	protected int crossOverRetryCountLimit;

	/**
	 * @param configFile
	 * @param crossOverRetryCountLimit
	 * @param mutationRetryCountLimit
	 * @param debugOn
	 */
	public abstract void intitPopulationStrategy(String configFile, int crossOverRetryCountLimit,  
			int mutationRetryCountLimit, boolean debugOn);

	/**
	 * @return
	 */
	public abstract PopulationSearchDomain createPopulationStrategyClone();


	/**
	 * @param firstSoln
	 * @param secondSoln
	 * @return
	 */
	public Pair<String, String> crossOverForPair(String firstSoln, String secondSoln) {
		boolean valid = false;
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
		for (int tryCount = 0; !valid && tryCount < crossOverRetryCountLimit; ++tryCount) {
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

	/**
	 * @param firstSoln
	 * @param secondSoln
	 * @return
	 */
	public String crossOverForOne(String firstSoln, String secondSoln) {
		String[] firstSolncomp = getSolutionComponenets(firstSoln);
		String[] secondSolncomp = getSolutionComponenets(secondSoln);
		String[] childSolncomp = new String[numComponents];
		Set<Integer> crossOverPoints = new HashSet<Integer>();
		boolean valid = false;
		
		//retry loop
		for (int tryCount = 0; !valid && tryCount < crossOverRetryCountLimit; ++tryCount) {
			int crossOverPt = BasicUtils.sampleUniform(1, numComponents-1);
			if (crossOverPoints.contains(crossOverPt)) {
				continue;
			} else {
				crossOverPoints.add(crossOverPt);
			}
			BasicUtils.arrayCopy(firstSolncomp, 0, crossOverPt, childSolncomp, 0);
			BasicUtils.arrayCopy(secondSolncomp, crossOverPt, numComponents, childSolncomp, crossOverPt);
			valid = isValid(childSolncomp);
		}
		String childSoln = aggregateSolutionComponenets(childSolncomp);
		if (!valid) {
			invalidSolutions.add(childSoln);
		}
		return childSoln;
	}	
}
