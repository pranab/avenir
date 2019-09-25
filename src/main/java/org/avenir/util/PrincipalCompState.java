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

package org.avenir.util;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chombo.util.BasicUtils;

/**
 * @author pranab
 *
 */
public class PrincipalCompState implements Serializable {
	private int dimension;
	private String key;
	private int numHiddenStates;
	private int count;
	private double visibleEnergy;
	private double[] hiddenEnergy;
	private double[] hiddenUnitEnergy;
	private double[][] princComps;
	
	/**
	 * 
	 */
	public PrincipalCompState() {
	}
	
	/**
	 * @param dimension
	 * @param key
	 * @param numHiddenStates
	 * @param visibleEnergy
	 * @param hiddenEnergy
	 * @param hiddenUnitEnergy
	 * @param weights
	 * @param delim
	 */
	public PrincipalCompState(String key, int dimension, int numHiddenStates, int count , double[] energy, 
			double[] hiddenUnitEnergy, double[][] princComps) {
		this.key = key;
		this.dimension = dimension;
		this.numHiddenStates = numHiddenStates;
		this.count = count;
		this.visibleEnergy = energy[0];
		this.hiddenEnergy = new double[numHiddenStates];
		System.arraycopy(energy, 1, hiddenEnergy, 0, numHiddenStates);
		this.hiddenUnitEnergy = hiddenUnitEnergy;
		this.princComps = princComps;
	}

	/**
	 * @param key
	 * @param dimension
	 * @param numHiddenStates
	 */
	public PrincipalCompState(String key, int dimension, int numHiddenStates) {
		this.key = key;
		this.dimension = dimension;
		this.numHiddenStates = numHiddenStates;
		count = 0;
		visibleEnergy = 0;
		hiddenEnergy = BasicUtils.createZeroFilledDoubleArray(numHiddenStates);
		hiddenUnitEnergy = BasicUtils.createDoubleArrayWithRandomValues(numHiddenStates, 0, 1);
		princComps = new double[numHiddenStates][dimension];
		for (int i = 0; i < numHiddenStates; ++i) {
			princComps[i] = BasicUtils.createOneHotDoubleArray(dimension, i);
		}
	}
	
	/**
	 * @return
	 */
	public int getDimension() {
		return dimension;
	}

	/**
	 * @return
	 */
	public String getKey() {
		return key;
	}

	/**
	 * @return
	 */
	public int getNumHiddenStates() {
		return numHiddenStates;
	}

	public int getCount() {
		return count;
	}

	/**
	 * @return
	 */
	public double getVisibleEnergy() {
		return visibleEnergy;
	}

	/**
	 * @return
	 */
	public double[] getHiddenEnergy() {
		return hiddenEnergy;
	}

	/**
	 * @return
	 */
	public double[] getHiddenUnitEnergy() {
		return hiddenUnitEnergy;
	}

	/**
	 * @return
	 */
	public double[][] getPrincComps() {
		return princComps;
	}

	/**
	 * @param lines
	 * @param offset
	 * @return
	 */
	public int deserialize(List<String> lines, int offset, String delim) {
		int i = offset;
		String[] items = BasicUtils.getTrimmedFields(lines.get(i++), delim);
		int j = 0;
		key = items[j++];
		numHiddenStates = Integer.parseInt(items[j++]);
		count = Integer.parseInt(items[j++]);
		visibleEnergy = Double.parseDouble(items[j++]);
		for (int k = 0; k < numHiddenStates; ++k) {
			hiddenEnergy[k] = Double.parseDouble(items[j++]);
		}
		items = BasicUtils.getTrimmedFields(lines.get(i++), delim);
		hiddenUnitEnergy = BasicUtils.toDoubleArray(items);
		
		princComps = new double[numHiddenStates][dimension];
		for (int k = 0; k < numHiddenStates; ++k, ++i) {
			items = BasicUtils.getTrimmedFields(lines.get(i++), delim);
			princComps[k] = BasicUtils.toDoubleArray(items);
		}
		return i;
	}
	
	/**
	 * @param delim
	 * @param precision
	 * @return
	 */
	public List<String> serialize(String delim, int precision) {
		List<String> serialized = new ArrayList<String>();
		
		StringBuilder stBld = new StringBuilder();
		stBld.append(key).append(delim).append(numHiddenStates).append(delim).append(count).append(delim).
			append(BasicUtils.formatDouble(visibleEnergy, precision)).append(delim).
			append(BasicUtils.join(hiddenEnergy, delim, precision));
		serialized.add(stBld.toString());
		
		serialized.add(BasicUtils.join(hiddenUnitEnergy, delim, precision));
		
		for (double[] prComp : princComps) {
			serialized.add(BasicUtils.join(prComp, delim));
		}
		
		return serialized;
	}

	/**
	 * @param filePath
	 */
	public static Map<String, PrincipalCompState> load(String filePath, String delim) {
		Map<String, PrincipalCompState> pcStates = new HashMap<String, PrincipalCompState>();
		
		try {
			List<String> lines = BasicUtils.getFileLines(filePath);
			int offset = 0;
			while (offset < lines.size()) {
				PrincipalCompState pcState = new PrincipalCompState();
				offset = pcState.deserialize(lines, offset, delim);
				pcStates.put(pcState.getKey(), pcState);
			}
		} catch (IOException e) {
			BasicUtils.assertFail("Failed to load PCA state " + e.getMessage());
		}
		return pcStates;
	}
	
}
