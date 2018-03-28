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

/**
 * @author pranab
 *
 */
public class LearningParameterSearch extends BasicSearchDomain {

	@Override
	public void initTrajectoryStrategy(String configFile, int maxStepSize,
			int mutationRetryCountLimit, boolean debugOn) {
		// TODO Auto-generated method stub

	}

	@Override
	public BasicSearchDomain createTrajectoryStrategyClone() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected void replaceSolutionComponent(String[] comonents, int index) {
		// TODO Auto-generated method stub

	}

	@Override
	public  double getSolutionCost(String solution) {
		double cost = 0;
		//TODO
		return cost;
	}
	
	@Override
	protected double calculateCost(String comp) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean isValid(String[] components) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isValid(String[] componentsex, int index) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	protected void addComponent(String[] componenets, int index) {
		// TODO Auto-generated method stub

	}

	@Override
	protected double getInvalidSolutionCost() {
		// TODO Auto-generated method stub
		return 0;
	}

}
