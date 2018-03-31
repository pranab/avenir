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


package org.avenir.optimize;

import java.io.Serializable;

import org.chombo.util.BasicUtils;
import org.chombo.util.Pair;

/**
 * @author pranab
 *
 */
public class SolutionWithCost extends Pair<String, Double> implements Comparable<SolutionWithCost>, Serializable {
	private Mutation mutation;
	private int iterationNum;

	/**
	 * @param solution
	 * @param cost
	 */
	public SolutionWithCost(String solution, double cost) {
		left = solution;
		right = cost;
	}
	
	/**
	 * @param solution
	 * @param cost
	 * @param mutation
	 */
	public SolutionWithCost(String solution, double cost, Mutation mutation) {
		this(solution, cost);
		this.mutation = mutation;
	}

	/**
	 * @param solution
	 * @param cost
	 * @param mutation
	 * @param iterationNum
	 */
	public SolutionWithCost(String solution, double cost, Mutation mutation, int iterationNum) {
		this(solution, cost, mutation);
		this.iterationNum = iterationNum;
	}

	/**
	 * @return
	 */
	public Mutation getMutation() {
		return mutation;
	}

	/**
	 * @param mutation
	 */
	public void setMutation(Mutation mutation) {
		this.mutation = mutation;
	}
	
	/**
	 * @return
	 */
	public int getIterationNum() {
		return iterationNum;
	}
	
	/**
	 * @param iterationNum
	 */
	public void setIterationNum(int iterationNum) {
		this.iterationNum = iterationNum;
	}

	/**
	 * @return
	 */
	public String getSolution() {
		return left;
	}

	/**
	 * @return
	 */
	public double getCost() {
		return right;
	}
	
	@Override
	public int compareTo(SolutionWithCost that) {
		return this.right.compareTo(that.right);
	}
	
	@Override
	public int hashCode() {
		return left.hashCode();
	}

	@Override
	public boolean equals(Object other) {
		SolutionWithCost that = (SolutionWithCost)other;
		return left.equals(that.left);
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	public String toString() {
		return left + "," + BasicUtils.formatDouble(right);
	}
	
}
