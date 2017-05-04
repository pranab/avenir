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

package org.avenir.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chombo.util.Pair;

/**
 * @author pranab
 *
 */
public class EnsemblePredictiveModel  extends PredictiveModel {
	private List<WeightedModel> models = new ArrayList<WeightedModel>();
	private Map<String, Double> votes = new HashMap<String, Double>();
	private List<VoteCount> sortedVotes = new ArrayList<VoteCount>();
	private double minOddsRatio = -1.0;
	
	public EnsemblePredictiveModel() {
		super();
	}
	
	/**
	 * @param minOdds
	 * @return
	 */
	public EnsemblePredictiveModel withMinOdds(double minOddsRatio) {
		this.minOddsRatio = minOddsRatio;
		return this;
	}

	/**
	 * @param model
	 */
	public void addModel(PredictiveModel model) {
		addModel(model, 1.0);
	}
	
	/**
	 * @param model
	 */
	public void addModel(PredictiveModel model, double weight) {
		models.add(new WeightedModel(model, weight));
	}
	
	/* (non-Javadoc)
	 * @see org.avenir.model.PredictiveModel#predict(java.lang.String[])
	 */
	@Override
	public String predict(String[] items) {
		if (models.size() % 2 == 0) {
			throw new IllegalStateException("neem odd number of models in ensemble");
		}
		
		//get votes
		votes.clear();
		for (WeightedModel weightedModel : models) {
			PredictiveModel model = weightedModel.getLeft();
			double weight = weightedModel.getRight();
			String predClass = model.predict(items);
			Double count = votes.get(predClass);
			if (null == count) {
				votes.put(predClass, weight);
			} else {
				votes.put(predClass, count+weight);
			}
		}
		
		//sort by vote count
		sortedVotes.clear();
		for (String prClass : votes.keySet()) {
			Double voteCount = votes.get(prClass);
			sortedVotes.add(new VoteCount(prClass, voteCount));
		}
		Collections.sort(sortedVotes);
		
		if (minOddsRatio > 1.0) {
			//null implies ambiguous
			double oddsRatio = sortedVotes.get(0).getRight() / sortedVotes.get(1).getRight();
			predClass = oddsRatio > minOddsRatio ? sortedVotes.get(0).getLeft() : null;
		} else {
			//select max vote
			predClass = sortedVotes.get(0).getLeft();
		}
		
		if (errorCountingEnabled) {
			countError();
		}
		
		return predClass;
	}

	@Override
	protected Pair<String, Double> predictClassProb(String[] items) {
		return null;
	}
	
	private static class WeightedModel extends Pair<PredictiveModel, Double> {
		public WeightedModel(PredictiveModel model, double weight) {
			super(model, weight);
		}
	}
	
	/**
	 * @author pranab
	 *
	 */
	private static class VoteCount extends Pair<String, Double> implements Comparable<VoteCount> {

		/**
		 * @param predClass
		 * @param voteCount
		 */
		public VoteCount(String predClass, double voteCount) {
			super(predClass, voteCount);
		}
		
		@Override
		public int compareTo(VoteCount that) {
			return that.right.compareTo(this.right);
		}
	}

}
