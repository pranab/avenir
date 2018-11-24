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
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import org.chombo.distance.AttributeDistanceSchema;
import org.chombo.distance.InterRecordDistance;
import org.chombo.stats.CategoricalHistogramStat;
import org.chombo.util.GenericAttributeSchema;

/**
 * @author pranab
 *
 */
public class Cluster implements Serializable {
	//private static final long serialVersionUID = 1L;
	private int numClusterInGroup;
	private int groupId;
	private int id;
	private String centroid;
	private String newCentroid;
	private double movement;
	private String status; 
	private String[] items;
	private double distance;
	private double avDistance;
	private double sumDist;
	private double sumDistSq;
	private int count;
    private Map<Integer, Double> numSums = new HashMap<Integer, Double>();
    private Map<Integer, CategoricalHistogramStat> catHist = new HashMap<Integer, CategoricalHistogramStat>();

    public Cluster() {
    }
    
	/**
	 * @param groupId
	 * @param id
	 * @param centroid
	 */
	public Cluster(int numClusterInGroup, int groupId, int id, String centroid) {
		this.numClusterInGroup = numClusterInGroup;
		this.groupId = groupId;
		this.id = id;
		this.centroid = centroid;
	} 
	
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
	 * @param centroid
	 * @param status
	 * @param delim
	 */
	public Cluster(String centroid, String status, String delim) {
		this.centroid = centroid;
        this.status = status;
        this.items = centroid.split(delim, -1);
	}

	public void intialize() {
		numSums.clear();
		catHist.clear();
		count = 0;
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
	public double findDistaneToCentroid(String[] record,  
			InterRecordDistance distanceFinder) throws IOException {
		distance =  distanceFinder.findDistance(items, record);
		return distance;
	}
	
	/**
	 * @param attrOrdinals
	 * @param schema
	 */
	public void initMembership(int[] attrOrdinals, GenericAttributeSchema schema) {
		numSums.clear();
		catHist.clear();
       	for (int attr : attrOrdinals) {
       		if (schema.areNumericalAttributes(attr)) {
       			numSums.put(attr, 0.0);
       		} else if (schema.areCategoricalAttributes(attr)) {
       			catHist.put(attr, new CategoricalHistogramStat());
       		} else {
       			throw new IllegalStateException("only numerical and categorical attribute allowed");
       		}
       	}
       	sumDist = 0;
       	sumDistSq = 0;
       	count = 0;
	}

	/**
	 * @param record
	 * @param distance
	 * @param schema
	 * @param attrDistSchema
	 * @param distanceFinder
	 * @throws IOException
	 */
	public void addMember(String[] record, double distance, GenericAttributeSchema schema, AttributeDistanceSchema attrDistSchema, 
			InterRecordDistance distanceFinder) throws IOException {

		for (int i = 0 ; i < record.length; ++i) {
			Double sum = numSums.get(i);
			CategoricalHistogramStat hist = catHist.get(i);
			if (null != sum) {
				sum += Double.parseDouble(record[i]);
				numSums.put(i, sum);
			} else if (null != hist) {
				hist.add(record[i]);
			} else {
				//attribute not included
			}
			
			sumDist += distance;
			sumDistSq += distance * distance;
			++count;
		}
	}
	
	/**
	 * @param that
	 * @return
	 */
	public Cluster merge(Cluster that) {
		//numerical
		for (int nAttr :  numSums.keySet()) {
			numSums.put(nAttr, numSums.get(nAttr) + that.numSums.get(nAttr));
		}

		//categorical
		for (int cAttr : catHist.keySet()) {
			CategoricalHistogramStat merged = catHist.get(cAttr).merge(that.catHist.get(cAttr));
			catHist.put(cAttr, merged);
		}
		
		sumDist += that.sumDist;
		sumDistSq += that.sumDistSq;
		count += that.count;
		return this;
	}

	/**
	 * @return
	 */
	public double getDistance() {
		return distance;
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#hashCode()
	 */
	public int hashCode() {
		int hCode = 3 * numClusterInGroup +  7 * groupId + 23 * id + 43 * centroid.hashCode();
		hCode = hCode < 0 ? -hCode : hCode;
		return hCode;
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	public boolean equals(Object obj ) {
		boolean isEqual = false;
		if (null != obj && obj instanceof Cluster){
			Cluster that = (Cluster)obj;
			isEqual =  numClusterInGroup == that.numClusterInGroup && groupId == that.groupId && 
					id == that.id && centroid.equals(that.centroid);
		}
		return isEqual;
	}
	
}
