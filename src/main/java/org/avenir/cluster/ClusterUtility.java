/*
 * avenir: Predictive analytic on Spark and Hadoop
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

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

public class ClusterUtility {
	
	/**
	 * @param centroids
	 * @param distMeas
	 * @return
	 */
	public static Cluster[] getClusterProperies(List<CentroidCluster<DoublePoint>> centroids, DistanceMeasure distMeas) {
		Cluster[] clusters = new Cluster[centroids.size()];
		int i = 0;
		for (CentroidCluster<DoublePoint> centroid : centroids) {
			double[] centPoint  = centroid.getCenter().getPoint();
			List<DoublePoint> members = centroid.getPoints();
			
			//members
			List<double[]> memPoints = new ArrayList<double[]>();
			for(DoublePoint dp : members)  {
				memPoints.add(dp.getPoint());
			}
			int count = memPoints.size();

			//average distance 
			double sum  = 0;
			double sumSq  = 0;
			for (double[] memPoint : memPoints) {
				double dist = distMeas.compute(centPoint, memPoint);
				sum += dist;
				sumSq += dist * dist;
			}
			double avDist = sum / count;		
			double sse = sumSq / count;		
			Cluster cluster = new Cluster(centPoint, avDist,  sse, count);
			clusters[i++] = cluster;
		}
		return clusters;
	}
	
	/**
	 * @param clusters
	 * @return
	 */
	public static double getAverageSse(Cluster[] clusters) {
		double sum = 0;
		for (Cluster cl : clusters) {
			sum += cl.getSse();
		}
		return sum / clusters.length;
	}
	
	

}
