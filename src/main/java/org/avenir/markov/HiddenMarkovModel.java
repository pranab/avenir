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

package org.avenir.markov;

import java.util.List;

import org.chombo.util.TabularData;
import org.chombo.util.Utility;

/**
 * Data for HMM
 * @author pranab
 *
 */
public class HiddenMarkovModel {
	private String[] states;
	private String[] observations;
	private TabularData stateTransitionProb;
	private TabularData stateObservationProb;
	private int[] intialStateProb;
	private int numStates;
	private int numObservations;
	private static final  String DELIM = ",";
	
	
	/**
	 * @param states
	 * @param observations
	 */
	public HiddenMarkovModel(List<String> lines) {
		int count = 0;
		states = lines.get(count++).split(DELIM);
		observations = lines.get(count++).split(DELIM);
		numStates = states.length;
		numObservations = observations.length;
		
		//state transition probablity
		stateTransitionProb = new TabularData(numStates, numStates);
		for (int i = 0; i < numStates; ++i) {
			stateTransitionProb.deseralizeRow(lines.get(count++), i);
		}
		
		//state observation probability
		stateObservationProb = new TabularData(numStates, numObservations);
		for (int i = 0; i < numStates; ++i) {
			stateObservationProb.deseralizeRow(lines.get(count++), i);
		}
		
		//initial state probility
		intialStateProb =  Utility.intArrayFromString(lines.get(count++), DELIM);
	}
	
	/**
	 * @param indx
	 * @return
	 */
	public int[] getDestStateProbility(int stateIndx) {
		return stateTransitionProb.getRow(stateIndx);
	}

	/**
	 * @param indx
	 * @return
	 */
	public int[] getObservationProbility(int observationIndx) {
		return stateObservationProb.getRow(observationIndx);
	}

	/**
	 * @param state
	 * @param observation
	 * @return
	 */
	public int getObservationProbility(int stateIndx, int observationIndx) {
		return stateObservationProb.get(stateIndx,  observationIndx);
	}

	/**
	 * @param observation
	 * @return
	 */
	public int getObservationIndex(String observation) {
		int indx = 0;
		for (String obs : observations) {
			if (obs.equals(observation)) {
				break;
			}
			++indx;
		}
		return indx;
	}
	
	/**
	 * @param indx
	 * @return
	 */
	public String getState(int indx) {
		return states[indx];
	}

	public int getNumStates() {
		return numStates;
	}

	public int getNumObservations() {
		return numObservations;
	}
}
