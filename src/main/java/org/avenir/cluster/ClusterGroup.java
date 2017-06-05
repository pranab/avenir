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

import org.chombo.distance.InterRecordDistance;

/**
 * @author pranab
 *
 */
public class ClusterGroup {
	private List<Cluster> clusters = new ArrayList<Cluster>();
	private double movementThreshold;
	private boolean active;
	public static final String STATUS_ACTIVE = "active";
	public static final String STATUS_STOPPED = "stopped";
	
	/**
	 * @param movementThreshold
	 */
	public ClusterGroup(double movementThreshold) {
		super();
		this.movementThreshold = movementThreshold;
	}

	/**
	 * @param centroid
	 * @param movement
	 * @param status
	 */
	public void addCluster(String centroid,  double movement, String status, String delim) {
		status = movement < movementThreshold ?  STATUS_STOPPED  :  status;
		clusters.add( new Cluster(centroid,  movement, status, delim));
	}

	/**
	 * 
	 */
	public void initialize() {
		active = false;
		for( Cluster cluster : clusters ) {
			if (cluster.status.equals(STATUS_ACTIVE)) {
				active = true;
				break;
			}
		}
	}
	
	/**
	 * @return
	 */
	public boolean isActive() {
		return active;
	}
	
	/**
	 * @param record
	 * @param distanceFinder
	 * @return
	 * @throws IOException 
	 */
	public Cluster findClosestCluster(String[] record,  InterRecordDistance distanceFinder) throws IOException {
		Cluster nearest = null;
		double minDist = Double.MAX_VALUE;
		for (Cluster cluster : clusters) {
			double dist =  cluster.findDistaneToCentroid(record,   distanceFinder);
			if (dist < minDist) {
				minDist = dist;
				nearest = cluster;
			}
		}
		return nearest;
	}

	/**
	 * @author pranab
	 *
	 */
	public static class Cluster {
		private String centroid;
		private double movement;
		private String status; 
		private String[] items;
		private double distance;
		
		/**
		 * @param centroid
		 * @param movement
		 * @param status
		 * @param delim
		 */
		public Cluster(String centroid,  double movement, String status, String delim) {
			this.centroid = centroid;
            this.movement = movement;
            this.status = status;
            this.items = centroid.split(delim, -1);
		}

		/**
		 * @return
		 */
		public String getCentroid() {
			return centroid;
		}

		/**
		 * @return
		 */
		public double getMovement() {
			return movement;
		}

		/**
		 * @return
		 */
		public String getStatus() {
			return status;
		}
		
		/**
		 * @param record
		 * @param distanceFinder
		 * @return
		 * @throws IOException
		 */
		public double findDistaneToCentroid(String[] record,  InterRecordDistance distanceFinder) throws IOException {
			distance =  distanceFinder.findDistance(items, record);
			return distance;
		}

		/**
		 * @return
		 */
		public double getDistance() {
			return distance;
		}
	}

}
