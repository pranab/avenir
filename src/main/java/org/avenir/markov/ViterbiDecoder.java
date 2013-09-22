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

import org.apache.log4j.Logger;
import org.chombo.util.DoubleTable;
import org.chombo.util.TabularData;

/**
 * State sequence predictor based on given observation sequence and HMM model
 * using viterbi algorithm
 * @author pranab
 *
 */
public class ViterbiDecoder {
	private DoubleTable statePathProb;
	private TabularData statePtr;
	private HiddenMarkovModel model;
	private boolean processed;
	private int numObs;
	private int curObsIndex;
	private int numStates;
	private static Logger LOG;
	
	/**
	 * @param model
	 */
	public ViterbiDecoder(HiddenMarkovModel model, Logger LOG) {
		ViterbiDecoder.LOG = LOG;
		this.model = model;
		numStates = model.getNumStates();
	}
	
	/**
	 * @param numObs
	 */
	public void initialize(int numObs) {
		this.numObs = numObs;
		statePathProb = new DoubleTable(numObs, numStates);
		statePtr = new TabularData(numObs, numStates);
		processed = false;
		curObsIndex = 0;
		LOG.debug("numObs:" + numObs);
	}
	
	/**
	 * process next observation
	 * @param observation
	 */
	public void nextObservation(String observation) {
		int obsIndx = model.getObservationIndex(observation);
		double stateProb, obsProb, pathProb, priorPathProb, transProb, maxPathProb;
		int maxPathProbStateIndx;
		LOG.debug("curObsIndex:" + curObsIndex);
		if (!processed) {
			//first use initial state probability
			for (int stateIndx = 0; stateIndx < numStates; ++stateIndx) {
				stateProb = model.getIntialStateProbability(stateIndx);
				obsProb = model.getObservationProbabiility(stateIndx, obsIndx);
				pathProb = stateProb * obsProb;
				LOG.debug("pathProb:" + pathProb );
				statePathProb.set(curObsIndex, stateIndx, pathProb);
				statePtr.set(curObsIndex, stateIndx, -1);
			}
			processed = true;
		} else {
			//iterative for subsequent using prevoious state path probability 
			for (int stateIndx = 0; stateIndx < numStates; ++stateIndx) {
				maxPathProb  = 0;
				maxPathProbStateIndx = 0;
				obsProb = model.getObservationProbabiility(stateIndx, obsIndx);
				for (int priorStateIndx = 0; priorStateIndx < numStates; ++priorStateIndx) {
					priorPathProb =statePathProb.get(curObsIndex-1, priorStateIndx);
					transProb = model.getDestStateProbility(priorStateIndx, stateIndx);
					pathProb = priorPathProb * transProb;
					
					if (pathProb > maxPathProb) {
						maxPathProb = pathProb;
						maxPathProbStateIndx = priorStateIndx;
					}
				}
				
				LOG.debug("maxPathProb:" + maxPathProb + " maxPathProbStateIndx:"  + maxPathProbStateIndx);
				statePathProb.set(curObsIndex, stateIndx, maxPathProb * obsProb);
				statePtr.set(curObsIndex, stateIndx, maxPathProbStateIndx);
			}			
		}
		++curObsIndex;
	}
	
	/**
	 * Get state sequence starting with latest
	 * @return
	 */
	public String[] getStateSequence() {
		String[] states = new String[numObs];
		int stateSeqIndx = 0;
		double pathProb;
		double maxPathProb = 0;
		int maxProbStateIndx = -1;
		int priorStateIndx;
		int nextStateIndx = -1;
		
		//state at end of observation sequence
		LOG.debug("state seq" );
		maxPathProb = 0;
		for (int stateIndx = 0; stateIndx < numStates; ++stateIndx) {
			//max path probability for the last observation
			pathProb = statePathProb.get(numObs -1, stateIndx);
			if (pathProb > maxPathProb) {
				maxPathProb = pathProb;
				maxProbStateIndx = stateIndx;
			}
		}
		LOG.debug("maxProbStateIndx:" + maxProbStateIndx);
		states[stateSeqIndx++] = model.getState(maxProbStateIndx);
		nextStateIndx = maxProbStateIndx;
		
		//backtrack for rest of the states going back ward
		for (int obsIndx = numObs -1 ; obsIndx >= 1; --obsIndx) {
			priorStateIndx = statePtr.get(obsIndx, nextStateIndx);
			LOG.debug("priorStateIndx:" + priorStateIndx);
			states[stateSeqIndx++] = model.getState(priorStateIndx);
			nextStateIndx = priorStateIndx;
		}
		return states;
	}
}
