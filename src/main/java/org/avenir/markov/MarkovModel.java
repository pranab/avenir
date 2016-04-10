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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chombo.util.DoubleTable;

/**
 * @author pranab
 *
 */
public class MarkovModel {
	private String[] states;
	private DoubleTable stateTransitionProb;
	private Map<String, DoubleTable> classBasedStateTransitionProb = new HashMap<String, DoubleTable>();
	private int numStates;
	
	private static final  String DELIM = ",";
	
	public MarkovModel(List<String> lines, boolean isClassLabelBased) {
		int count = 0;
		states = lines.get(count++).split(DELIM);
		numStates = states.length;
		if (isClassLabelBased) {
			String curClassLabel = null;
			DoubleTable curStateTransitionProb = null;
			while (count < lines.size()) {
				String line = lines.get(count);
				if (line.startsWith("classLabel")) {
					curClassLabel = line.split(":")[1];
					++count;
				} else {
					curStateTransitionProb = new DoubleTable(states, states);
					for (int i = 0; i < numStates; ++i) {
						curStateTransitionProb.deseralizeRow(lines.get(count), i);
						++count;
					}
					classBasedStateTransitionProb.put(curClassLabel, curStateTransitionProb);
				}
			}
		} else {
			stateTransitionProb = new DoubleTable(states, states);
			for (int i = 0; i < numStates; ++i) {
				stateTransitionProb.deseralizeRow(lines.get(count++), i);
			}
		}
	}	
	
	/**
	 * @param rowState
	 * @param colState
	 * @return
	 */
	public double getStateTransProbability(String rowState, String colState) {
		return stateTransitionProb.get(rowState, colState);
	}
	
	/**
	 * @param classLabel
	 * @param rowState
	 * @param colState
	 * @return
	 */
	public double getStateTransProbability(String classLabel, String rowState, String colState) {
		DoubleTable stateTransitionProb = classBasedStateTransitionProb.get(classLabel);
		return stateTransitionProb.get(rowState, colState);
	}
	
}
