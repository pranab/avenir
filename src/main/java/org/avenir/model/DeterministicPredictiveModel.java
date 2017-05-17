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

import org.chombo.util.FeatureSchema;

/**
 * Predictor for models that make deterministic prediction
 * @author pranab
 *
 */
public abstract class DeterministicPredictiveModel  extends PredictiveModel{

	/**
	 * @param schema
	 */
	public DeterministicPredictiveModel(FeatureSchema schema) {
		super(schema);
	}

	/**
	 * @param items
	 * @return
	 */
	@Override
	public String predict(String[] items) {
		predictClassProb(items);
		predClass = predClassProb.getLeft();
		
		if (errorCountingEnabled) {
			countError();
		}
		return predClass;
	}

}
