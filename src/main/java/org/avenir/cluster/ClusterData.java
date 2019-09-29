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
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chombo.distance.AttributeDistanceSchema;
import org.chombo.distance.InterRecordDistance;
import org.chombo.stats.CategoricalHistogramStat;
import org.chombo.util.BasicUtils;
import org.chombo.util.GenericAttributeSchema;

/**
 * @author pranab
 *
 */
public class ClusterData implements Serializable {
	//private static final long serialVersionUID = 1L;
	private int numClusterInGroup;
	private int groupId;
	private int id;
	private String centroid;
	private String newCentroid;
	private double movement;
	private boolean active; 
	private String[] items;
	private double distance;
	private double avDistance;
	private double sumDist;
	private double sumDistSq;
	private int count;
	public double sse;
    private Map<Integer, Double> numSums = new HashMap<Integer, Double>();
    private Map<Integer, CategoricalHistogramStat> catHist = new HashMap<Integer, CategoricalHistogramStat>();
    private int outputPrecision = 3;
    private String fieldDelim;

    public ClusterData() {
    }
    
	/**
	 * @param groupId
	 * @param id
	 * @param centroid
	 */
	public ClusterData(int numClusterInGroup, int groupId, int id, String centroid, String delim) {
		this.numClusterInGroup = numClusterInGroup;
		this.groupId = groupId;
		this.id = id;
		this.centroid = centroid;
		this.items = centroid.split(delim, -1);
	} 
	
	/**
	 * @param centroid
	 * @param movement
	 * @param status
	 * @param delim
	 */
	public ClusterData(String centroid,  double movement, boolean active, String delim) {
		this.centroid = centroid;
        this.movement = movement;
        this.active = active;
        this.items = centroid.split(delim, -1);
	}

	/**
	 * @param centroid
	 * @param status
	 * @param delim
	 */
	public ClusterData(String centroid, boolean active, String delim) {
		this.centroid = centroid;
        this.active = active;
        this.items = centroid.split(delim, -1);
	}

	/**
	 * 
	 */
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
	public boolean isActive() {
		return active;
	}
	
	/**
	 * @return
	 */
	public int getNumClusterInGroup() {
		return numClusterInGroup;
	}

	/**
	 * @return
	 */
	public int getGroupId() {
		return groupId;
	}

	/**
	 * @return
	 */
	public int getId() {
		return id;
	}

	/**
	 * @return
	 */
	public double getAvDistance() {
		return avDistance;
	}

	/**
	 * @return
	 */
	public double getSse() {
		return sse;
	}

	/**
	 * @param outputPrecision
	 * @return
	 */
	public ClusterData withOutputPrecision(int outputPrecision) {
		this.outputPrecision = outputPrecision;
		return this;
	}

	/**
	 * @param fieldDelim
	 * @return
	 */
	public ClusterData withFieldDelim(String fieldDelim) {
		this.fieldDelim = fieldDelim;
		return this;
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
	public ClusterData merge(ClusterData that) {
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
	 * @param previous
	 * @param distanceFinder
	 * @param outputPrecision
	 * @param delim
	 * @throws IOException
	 */
	public void finishMemebership(ClusterData previous, InterRecordDistance distanceFinder, double centroidShiftThreshold, 
			int outputPrecision, String delim) throws IOException {
		numClusterInGroup = previous.numClusterInGroup;
		groupId = previous.groupId;
		id = previous.id;
		String[] pFields = BasicUtils.getTrimmedFields(previous.centroid, delim);		
		
		sse = sumDistSq / count;
		avDistance = sumDist / count;
		
		//numerical attr mean
		List<Integer> attrs = new ArrayList<Integer>(numSums.keySet());
		for (int attr : attrs) {
			double sum = numSums.get(attr);
			numSums.put(attr, sum/count);
		}
		
		//new centroid
		int recSize = pFields.length;
		String[] newCentroid = new String[recSize];
		for (int i = 0; i < recSize; ++i) {
			Double sum = numSums.get(i);
			CategoricalHistogramStat hist = catHist.get(i);
			if (null != sum) {
				newCentroid[i] = BasicUtils.formatDouble(sum,outputPrecision);
			} else if (null != hist) {
				newCentroid[i] = hist.getMode();
			} else {
				//attribute not included
				newCentroid[i] = null;
			}
		}
		centroid = BasicUtils.join(newCentroid, delim);
		
		//movement
		movement =  distanceFinder.findDistance(newCentroid, pFields);
		active = movement > centroidShiftThreshold;
	}
	
	public void makeCurrent(ClusterData previous, InterRecordDistance distanceFinder) throws IOException {
		numClusterInGroup = previous.numClusterInGroup;
		groupId = previous.groupId;
		id = previous.id;
		
		//movement
		String[] fields = BasicUtils.getTrimmedFields(centroid, ",");
		String[] pFields = BasicUtils.getTrimmedFields(previous.centroid, ",");
		movement =  distanceFinder.findDistance(fields, pFields);
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
		if (null != obj && obj instanceof ClusterData){
			ClusterData that = (ClusterData)obj;
			isEqual =  numClusterInGroup == that.numClusterInGroup && groupId == that.groupId && 
					id == that.id && centroid.equals(that.centroid);
		}
		return isEqual;
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		StringBuilder stBld = new StringBuilder();
		String sseStr = BasicUtils.formatDouble(sse, outputPrecision);
		String avDistanceStr = BasicUtils.formatDouble(avDistance, outputPrecision);
		stBld.append(numClusterInGroup).append(fieldDelim).append(groupId).
			append(fieldDelim).append(sseStr).append(fieldDelim).append(avDistanceStr).
			append(fieldDelim).append(centroid);
		return stBld.toString();
	}
}
