/*
 * avenir-spark: Predictive analytic based on Spark
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

package org.avenir.optimize;

import java.io.Serializable;

/**
 * @author pranab
 *
 */
public class Mutation implements Serializable {
	private int type;
	private String mutation;
	private int iterationNum;
	
	public Mutation(int type, String mutation) {
		this.type = type;
		this.mutation = mutation;
	}
	
	public Mutation(int type, String mutation, int iterationNum) {
		this(type,  mutation);
		this.iterationNum = iterationNum;
	}
	
	public int getType() {
		return type;
	}
	public void setType(int type) {
		this.type = type;
	}
	public String getMutation() {
		return mutation;
	}
	public void setMutation(String mutation) {
		this.mutation = mutation;
	}
	public int getIterationNum() {
		return iterationNum;
	}
	public void setIterationNum(int iterationNum) {
		this.iterationNum = iterationNum;
	}
	
}
