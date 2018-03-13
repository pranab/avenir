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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author pranab
 *
 */
public abstract  class TabuSearchDomain extends BasicSearchDomain {
	protected List<Mutation> tabuList = new ArrayList<Mutation>();
	protected int tabuTenure;
	protected List<SolutionWithCost> solutions = new ArrayList<SolutionWithCost>();
	protected List<SolutionWithCost> bestSolutions = new ArrayList<SolutionWithCost>();
	protected Set<SolutionWithCost> tabuViolated = new HashSet<SolutionWithCost>();
	
	public int getTabuTenure() {
		return tabuTenure;
	}

	public void setTabuTenure(int tabuTenure) {
		this.tabuTenure = tabuTenure;
	}
	
	/**
	 * @param mutation
	 */
	public void addTabu(Mutation mutation) {
		tabuList.add(mutation);
	} 
	
	/**
	 * @param curIterationNum
	 */
	public void purge(int curIterationNum) {
		List<Mutation> newTabuList = new ArrayList<Mutation>();
		for (Mutation mutation : tabuList) {
			if (curIterationNum - mutation.getIterationNum() <= tabuTenure) {
				newTabuList.add(mutation);
			}
		}
		
		if (newTabuList.size() < tabuList.size()) {
			tabuList.clear();
			tabuList.addAll(newTabuList);
		}
	}
	
	@Override
	public void prepareMutateSolution() {
		solutions.clear();
		tabuViolated.clear();
	}
}
