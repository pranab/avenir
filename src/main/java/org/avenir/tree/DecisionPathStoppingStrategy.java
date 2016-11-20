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


package org.avenir.tree;

import org.avenir.util.InfoContentStat;

/**
 * Strategies for implementing decision path termination
 * @author pranab
 *
 */
public class DecisionPathStoppingStrategy {
    private String stoppingStrategy;
    private int maxDepthLimit = -1;
    private double minInfoGainLimit = -1;
    private int minPopulationLimit = -1;
    public static final String STOP_MAX_DEPTH = "maxDepth";
    public static final String STOP_MIN_POPULATION = "minPopulation";
    public static final String STOP_MIN_INFO_GAIN = "minInfoGain";
    
	/**
	 * @param stoppingStrategy
	 * @param maxDepthLimit
	 * @param minInfoGainLimit
	 * @param minPopulationLimit
	 */
	public DecisionPathStoppingStrategy(String stoppingStrategy, int maxDepthLimit, 
		double minInfoGainLimit, int minPopulationLimit) {
		this.stoppingStrategy = stoppingStrategy;
		this.maxDepthLimit = maxDepthLimit;
		this.minInfoGainLimit = minInfoGainLimit;
		this.minPopulationLimit = minPopulationLimit;
	}

	/**
	 * @param stat
	 * @param parentStat
	 * @param currentDepth
	 * @return
	 */
	public boolean  shouldStop(InfoContentStat stat, double parentStat, int currentDepth) {
		boolean toBeStopped = false;
		if (stoppingStrategy.equals(STOP_MIN_POPULATION)) {
			toBeStopped = stat.getTotalCount() < minPopulationLimit;
		} else if (stoppingStrategy.equals(STOP_MIN_INFO_GAIN)) {
			toBeStopped = (parentStat - stat.getStat()) < minInfoGainLimit;
		} else if (stoppingStrategy.equals(STOP_MAX_DEPTH)) {
			toBeStopped = currentDepth >= maxDepthLimit;
		} else {
			throw new IllegalArgumentException("invalid stopping strategy " + stoppingStrategy);
		}
		
		return toBeStopped;
	}
    
}
