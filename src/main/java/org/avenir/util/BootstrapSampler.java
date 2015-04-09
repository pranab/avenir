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

package org.avenir.util;

/**
 * Does boot strap sampling without replacement. Same record may be included multiple times
 * and some records many not be sampled
 * @author pranab
 *
 */
public class BootstrapSampler {
	private String[] samples;
	private int sampleSize;
	private int curSize;
	private int sampleIter;
	
	/**
	 * @param sampleSize
	 */
	public BootstrapSampler(int sampleSize) {
		super();
		this.sampleSize = sampleSize;
		samples = new String[sampleSize];
		curSize = sampleSize;
		initialize();
	}
	
	/**
	 * @return
	 */
	public boolean isFull() {
		return curSize == sampleSize;
	}
	
	/**
	 * @param record
	 */
	public void add(String record) {
		samples[curSize++] = record;
	}
	
	/**
	 * 
	 */
	public void startSampling() {
		sampleIter = 0;
	}
	
	/**
	 * @return true if there are more samples
	 */
	public boolean hasSamples(){
		return sampleIter < curSize;
	}
	
	/**
	 * @return next sample
	 */
	public String nextSample() {
		int sel = (int)(Math.random() * curSize);
		++sampleIter;
		return samples[sel];
	}
	
	/**
	 * 
	 */
	public void initialize() {
		for (int i = 0; i < curSize; ++i) {
			samples[i] = null;
		}
		curSize = 0;
	}
	
}
