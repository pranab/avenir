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

package org.avenir.cluster;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.avenir.util.EntityDistanceMapFileAccessor;

/**
 * @author pranab
 *
 */
public class EdgeWeightedCluster {
	List<String> cluster = new ArrayList<String>();
	private double avEdgeWeight;
	private double distScale;
	private boolean isDistance;
	
	/**
	 * @param entityId
	 */
	public void add(String entityId) {
		cluster.add(entityId);
	}
	
	/**
	 * @param entityId
	 * @param distanceMapStorage
	 * @return
	 * @throws IOException
	 */
	public double tryMembership(String entityId, EntityDistanceMapFileAccessor distanceMapStorage) 
		throws IOException {
		double  newAvEdgeWeight = 0;
		double weightSum = 0;
		for (String memeberId :  cluster) {
			Map<String, Double> distances = distanceMapStorage.read(memeberId);
			Double dist = distances.get(entityId);
			if (null != dist) {
				if (isDistance) {
					weightSum += (distScale - dist);
				} else {
					weightSum +=  dist;
				}
			} 
		}
		int clusterSize = cluster.size();
		int numEdges = (clusterSize * (clusterSize -1)) / 2;
		newAvEdgeWeight = (avEdgeWeight * numEdges + weightSum) / (numEdges + clusterSize);
		return newAvEdgeWeight;
	}

	public void setDistScale(double distScale) {
		this.distScale = distScale;
		isDistance = true;
	}
	
}
