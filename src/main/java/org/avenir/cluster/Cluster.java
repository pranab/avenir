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
	protected double[] numCentroid;
	protected double avDistance;
	protected double sse;
	protected int count;
	protected int outputPrecision = 3;
	protected String fieldDelim;
	protected String centroidDelim = " ";

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
	public Cluster(double[] numCentroid, double avDistance, double sse, int count) {
		super();
		this.numCentroid = numCentroid;
		this.avDistance = avDistance;
		this.sse = sse;
		this.count = count;
	}

	public double[] getNumCentroid() {
		return numCentroid;
	}

	public void setNumCentroid(double[] numCentroid) {
		this.numCentroid = numCentroid;
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
		StringBuilder stBld = new StringBuilder(BasicUtils.join(numCentroid, centroidDelim, outputPrecision));
			stBld.append(fieldDelim).append(count).append(BasicUtils.formatDouble(avDistance, outputPrecision)).
			append(fieldDelim).append(BasicUtils.formatDouble(sse, outputPrecision));
		
		return stBld.toString();
	}
}
