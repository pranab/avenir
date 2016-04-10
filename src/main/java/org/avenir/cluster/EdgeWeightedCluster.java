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
import java.util.UUID;

import org.avenir.util.EntityDistanceMapFileAccessor;

/**
 * @author pranab
 *
 */
public class EdgeWeightedCluster {
	private List<String> cluster = new ArrayList<String>();
	private String ID;
	private double avEdgeWeight;
	private double distScale;
	private boolean isDistance;
	private String delim = ",";
	
	public EdgeWeightedCluster() {
		ID = UUID.randomUUID().toString().replaceAll("-", "");
	}
	
	/**
	 * @param entityId
	 */
	public void add(String entityId, double avEdgeWeight) {
		cluster.add(entityId);
		this.avEdgeWeight = avEdgeWeight;
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

	/**
	 * @param distScale
	 */
	public void setDistScale(double distScale) {
		this.distScale = distScale;
		isDistance = true;
	}

	/**
	 * @return
	 */
	public String getID() {
		return ID;
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		StringBuilder stBld = new StringBuilder();
		stBld.append(ID).append(delim);
		for (String entity :  cluster) {
			stBld.append(entity).append(delim);
		}
		stBld.append(avEdgeWeight);
		return stBld.toString();
	}
	
}
