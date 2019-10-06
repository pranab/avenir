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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.chombo.util.BasicUtils;
import org.chombo.util.Record;

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
	
	/**
	 * @param clusters
	 * @return
	 */
	public static double getAverageSse(ClusterData[] clusters) {
		double sum = 0;
		for (Cluster cl : clusters) {
			sum += cl.getSse();
		}
		return sum / clusters.length;
	}
	
	/**
	 * @param filePath
	 * @param delim
	 * @param key Len
	 * @return
	 */
	public static Map<String, List<ClusterData>> load(String filePath, String delim, int keyLen) {
		Map<String, List<ClusterData>> keyedClusters = new HashMap<String, List<ClusterData>>();
		try {
			List<String> lines = BasicUtils.getFileLines(filePath);
			for (String line : lines) {
				List<ClusterData> clusters = new ArrayList<ClusterData>();
	    		int pos = BasicUtils.findOccurencePosition(line, delim, keyLen, true);
	    		String key = line.substring(0, pos);
	    		
	    		Record rec = new Record(line.substring(pos), delim);
	    		while (rec.hasNext()) {
		    		String centroid = rec.getString();
		    		centroid.replaceAll(" ", delim);
		    		int count = rec.getInt();
		    		double avDist = rec.getDouble();
		    		double sse = rec.getDouble();;
		    		ClusterData clData = new ClusterData(centroid,  count, avDist, sse, delim); 
		    		clusters.add(clData);
	    		}
	    		keyedClusters.put(key, clusters);
			}
		} catch (IOException e) {
			BasicUtils.assertFail("failed to open cluster definition file " + e.getMessage());
		}
		
		return keyedClusters;
	}
	
	/**
	 * @param clusters
	 * @param largeClusterSizeFraction
	 * @param largeClusterSizeMultilier
	 * @return
	 */
	public static List<ClusterData> labelSize(List<ClusterData> clusters, double largeClusterSizeFraction,
			double largeClusterSizeMultilier) {
		List<ClusterData> labeledClusters = new ArrayList<ClusterData>();
		
		return labeledClusters;
	}

}
