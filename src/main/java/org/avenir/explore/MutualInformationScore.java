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

package org.avenir.explore;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.chombo.mr.FeatureField;
import org.chombo.util.Pair;
import org.chombo.util.Triplet;

/**
 * Processes mutual info score
 * @author pranab
 *
 */
public class MutualInformationScore {
	private List<FeatureMutualInfo>  featureClassMutualInfoList = new ArrayList<FeatureMutualInfo>();
	private List<FeaturePairMutualInfo> featurePairMutualInfoList = new ArrayList<FeaturePairMutualInfo>();
	private List<FeaturePairMutualInfo> featurePairClassMutualInfoList = new ArrayList<FeaturePairMutualInfo>();
	
	/**
	 * @author pranab
	 *
	 */
	public  static class FeatureMutualInfo extends  Pair<Integer, Double>  implements  Comparable<FeatureMutualInfo> {
		public FeatureMutualInfo( int featureOrdinal, double mutualInfo) {
			super( featureOrdinal, mutualInfo);
		}
		
		@Override
		public int compareTo(FeatureMutualInfo that) {
			return that.getRight().compareTo(this.getRight());
		}
	}
	
	/**
	 * @author pranab
	 *
	 */
	public static class FeaturePairMutualInfo extends Triplet<Integer, Integer, Double> {
		public FeaturePairMutualInfo(int firstFeatureOrdinal, int secondFeatureOrdinal, double mutualInfo) {
			super(firstFeatureOrdinal, secondFeatureOrdinal, mutualInfo);
		}
	}
	
	/**
	 * @param featureOrdinal
	 * @param mutualInfo
	 */
	public void addFeatureClassMutualInfo(int featureOrdinal, double mutualInfo) {
		FeatureMutualInfo featureClassMutualInfo = new FeatureMutualInfo( featureOrdinal, mutualInfo);
		featureClassMutualInfoList.add(featureClassMutualInfo);
	}
	
	/**
	 * 
	 */
	public void sortFeatureMutualInfo() {
		Collections.sort(featureClassMutualInfoList);
	}

	/**
	 * Mutual lInformation Maximization (MIM)
	 * @return
	 */
	public List<FeatureMutualInfo> getMutualInfoMaximizerScore() {
		sortFeatureMutualInfo();
		return featureClassMutualInfoList;
	}

	/**
	 * @param featureOrdinal
	 * @param mutualInfo
	 */
	public void addFeaturePairMutualInfo(int firstFeatureOrdinal, int secondFeatureOrdinal, double mutualInfo) {
		FeaturePairMutualInfo featurepairMutualInfo = new FeaturePairMutualInfo( firstFeatureOrdinal, secondFeatureOrdinal, mutualInfo);
		featurePairMutualInfoList.add(featurepairMutualInfo);
	}
	
	/**
	 * Mutual Information Feature Selection (MIFS)
	 * @return
	 */
	public List<FeatureMutualInfo> getMutualInfoFeatureSelectionScore(double redunacyFactor) {
		List<FeatureMutualInfo>  mutualInfoFeatureSelection = new ArrayList<FeatureMutualInfo>();
		Set<Integer> selectedFeatures = new HashSet<Integer>();
		
		while (selectedFeatures.size() < featureClassMutualInfoList.size() ) {
			double maxScore = Double.NEGATIVE_INFINITY;
			int selectedFeature = 0;
			//all features
			for (FeatureMutualInfo muInfo :  featureClassMutualInfoList) {
				int feature = muInfo.getLeft();
				if (selectedFeatures.contains(feature)) {
					continue;
				}
				
				//all feature pair mutual info
				double sum = 0;
				for (FeaturePairMutualInfo  otherMuInfo :  featurePairMutualInfoList) {
					//pair with feature already selected
					if ( otherMuInfo.getLeft() == feature && selectedFeatures.contains(otherMuInfo.getCenter())) {
						sum +=  otherMuInfo.getRight();
					} else if ( otherMuInfo.getCenter() == feature && selectedFeatures.contains(otherMuInfo.getLeft())) {
						sum +=  otherMuInfo.getRight();
					} 
				}
				double score = muInfo.getRight() - redunacyFactor * sum;
				if (score > maxScore) {
					maxScore = score;
					selectedFeature = feature;
				}
			}
			
			//add the feature with max score
			FeatureMutualInfo featureClassMutualInfo = new FeatureMutualInfo( selectedFeature, maxScore);
			mutualInfoFeatureSelection.add(featureClassMutualInfo);
			selectedFeatures.add(selectedFeature);
		}
		return mutualInfoFeatureSelection;
	}
	
	/**
	 * @param featureOrdinal
	 * @param mutualInfo
	 */
	public void addFeaturePairClassMutualInfo(int firstFeatureOrdinal, int secondFeatureOrdinal, double mutualInfo) {
		FeaturePairMutualInfo featurepairMutualInfo = new FeaturePairMutualInfo( firstFeatureOrdinal, secondFeatureOrdinal, mutualInfo);
		featurePairClassMutualInfoList.add(featurepairMutualInfo);
	}
	
	/**
	 * Joint Mutual Info (JMI)
	 * @param featureFields
	 * @return
	 */
	public List<FeatureMutualInfo> getJointMutualInfoScore( List<FeatureField> featureFields) {
		List<FeatureMutualInfo>  featureJointMutualInfoList  = new ArrayList<FeatureMutualInfo>();
		Double score;
		Map <Integer, Double > jointMutualInfo = new HashMap <Integer, Double >();
		
		//all features
		for (FeatureField field : featureFields ) {
			int fieldOrd = field.getOrdinal();
			score = 0.0;
			jointMutualInfo.put(fieldOrd, score);
			for (FeaturePairMutualInfo featurePairMuInfo :  featurePairClassMutualInfoList) {
				//if paired with another feature
				if (featurePairMuInfo.getLeft() == fieldOrd || featurePairMuInfo.getCenter() == fieldOrd) {
					 score = jointMutualInfo.get(fieldOrd) + featurePairMuInfo.getRight();
					jointMutualInfo.put(fieldOrd, score );
				}
			}
		}
		
		//collect in a list and sort
		for (Integer feature : jointMutualInfo.keySet()) {
			FeatureMutualInfo featureJointMutualInfo = new FeatureMutualInfo( feature, jointMutualInfo.get(feature));
			featureJointMutualInfoList.add(featureJointMutualInfo);
		}
		Collections.sort(featureJointMutualInfoList);

		return featureJointMutualInfoList;
	}

	/**
	 * Min redundancy Max Relevance (MRMR)
	 * @return
	 */
	public List<FeatureMutualInfo> getMinRedundancyMaxrelevanceScore( ) {
		List<FeatureMutualInfo>  minRedundancyMaxrelevance  = new ArrayList<FeatureMutualInfo>();
		Set<Integer> selectedFeatures = new HashSet<Integer>();

		while (selectedFeatures.size() < featureClassMutualInfoList.size() ) {
			double maxScore = Double.NEGATIVE_INFINITY;
			int selectedFeature = 0;
			for (FeatureMutualInfo featureMuInfo : featureClassMutualInfoList) {
				int feature = featureMuInfo.getLeft();
				if (selectedFeatures.contains(feature)) {
					continue;
				}
				double feMuInfo = featureMuInfo.getRight();
				double sum = 0;
				for (FeaturePairMutualInfo featurePairMuInfo : featurePairMutualInfoList) {
					//pair with feature already selected
					if ( featurePairMuInfo.getLeft() == feature && selectedFeatures.contains(featurePairMuInfo.getCenter())) {
						sum +=  featurePairMuInfo.getRight();
					} else if ( featurePairMuInfo.getCenter() == feature && selectedFeatures.contains(featurePairMuInfo.getLeft())) {
						sum +=  featurePairMuInfo.getRight();
					} 
				}
				
				double score =   selectedFeatures.size() > 0 ?  feMuInfo - sum / selectedFeatures.size() : feMuInfo ;
				if (score > maxScore) {
					maxScore = score;
					selectedFeature = feature;
				}
			}
			//add the feature with max score
			FeatureMutualInfo featureClassMutualInfo = new FeatureMutualInfo( selectedFeature, maxScore);
			minRedundancyMaxrelevance.add(featureClassMutualInfo);
			selectedFeatures.add(selectedFeature);
		}
		return minRedundancyMaxrelevance;
	}
	
}
