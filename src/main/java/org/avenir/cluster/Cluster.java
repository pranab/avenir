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

import java.io.Serializable;

import org.chombo.util.BasicUtils;

/**
 * @author pranab
 *
 */
public class Cluster implements Serializable {
	private double[] centroid;
	private double avDistance;
	public double sse;
	private int count;
    private int outputPrecision = 3;
    private String fieldDelim;

	/**
	 * 
	 */
	public Cluster() {
		
	}

	/**
	 * @param centroid
	 * @param avDistance
	 * @param count
	 * @param sse
	 */
	public Cluster(double[] centroid, double avDistance, double sse, int count) {
		super();
		this.centroid = centroid;
		this.avDistance = avDistance;
		this.sse = sse;
		this.count = count;
	}

	public double[] getCentroid() {
		return centroid;
	}

	public void setCentroid(double[] centroid) {
		this.centroid = centroid;
	}

	public double getAvDistance() {
		return avDistance;
	}

	public void setAvDistance(double avDistance) {
		this.avDistance = avDistance;
	}

	public int getCount() {
		return count;
	}

	public void setCount(int count) {
		this.count = count;
	}

	public double getSse() {
		return sse;
	}

	public void setSse(double sse) {
		this.sse = sse;
	}

	/**
	 * @param fieldDelim
	 * @return
	 */
	public Cluster withFieldDelim(String fieldDelim) {
		this.fieldDelim = fieldDelim;
		return this;
	}

	/**
	 * @param outputPrecision
	 * @return
	 */
	public Cluster withOutputPrecision(int outputPrecision) {
		this.outputPrecision = outputPrecision;
		return this;
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		StringBuilder stBld = new StringBuilder(BasicUtils.join(centroid, fieldDelim, outputPrecision));
			stBld.append(fieldDelim).append(count).append(BasicUtils.formatDouble(avDistance, outputPrecision)).
			append(count).append(BasicUtils.formatDouble(sse, outputPrecision));
		
		return stBld.toString();
	}
}
