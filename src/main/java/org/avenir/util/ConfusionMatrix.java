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

public class ConfusionMatrix {
	private String posClass;
	private String negClass;
	private int truePos;
	private int falsePos;
	private int trueNeg;
	private int falseNeg;
	
	public ConfusionMatrix( String negClass, String posClass) {
		this.negClass = negClass;
		this.posClass = posClass;
	}
	
	public void report(String predClass, String actualClass) {
		if (predClass.equals(posClass)) {
			if (actualClass.equals(posClass)) {
				++truePos;
			} else {
				++falsePos;
			}
		} else {
			if (actualClass.equals(negClass)) {
				++trueNeg;
			} else {
				++falseNeg;
			}
		}
	}

	public int getTruePos() {
		return truePos;
	}

	public int getFalsePos() {
		return falsePos;
	}

	public int getTrueNeg() {
		return trueNeg;
	}

	public int getFalseNeg() {
		return falseNeg;
	}
	
	public int getRecall() {
		return (100 * truePos) /(truePos + falseNeg);
	}
	
	public int getPrecision() {
		return  (100 * truePos) /(truePos + falsePos);
	}
	
	public int getAccuracy() {
		return (100 * (truePos + trueNeg)) / (truePos + trueNeg + falsePos + falseNeg );
	}

}
