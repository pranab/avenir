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

import org.chombo.util.FeatureField;
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
	private List<FeaturePairEntropy> featurePairClassEntropyList = new ArrayList<FeaturePairEntropy>();
	
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
	 * @author pranab
	 *
	 */
	public static class FeaturePairEntropy extends Triplet<Integer, Integer, Double> {
		public FeaturePairEntropy(int firstFeatureOrdinal, int secondFeatureOrdinal, double entropy) {
			super(firstFeatureOrdinal, secondFeatureOrdinal, entropy);
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
		FeaturePairMutualInfo featurePairMutualInfo = new FeaturePairMutualInfo( firstFeatureOrdinal, secondFeatureOrdinal, mutualInfo);
		featurePairClassMutualInfoList.add(featurePairMutualInfo);
	}
	
	/**
	 * @param featureOrdinal
	 * @param mutualInfo
	 */
	public void addFeaturePairClassEntropy(int firstFeatureOrdinal, int secondFeatureOrdinal, double entropy) {
		FeaturePairEntropy featurePairEntropy = new FeaturePairEntropy( firstFeatureOrdinal, secondFeatureOrdinal, entropy);
		featurePairClassEntropyList.add(featurePairEntropy);
	}

	/**
	 * Joint Mutual Info (JMI)
	 * @return
	 */
	public List<FeatureMutualInfo> getJointMutualInfoScore() {
		return getJointMutualInfoScoreHelper(true);
	}

	/**
	 * Double Input Symetrical Relevance  (DISR)
	 * @return
	 */
	public List<FeatureMutualInfo> getDoubleInputSymmetricalRelevanceScore() {
		return getJointMutualInfoScoreHelper(false);
	}
	
	/**
	 * Joint Mutual Info (JMI)
	 * @param featureFields
	 * @return
	 */
	private List<FeatureMutualInfo> getJointMutualInfoScoreHelper(boolean joinMutInfo ) {
		List<FeatureMutualInfo>  featureJointMutualInfoList  = new ArrayList<FeatureMutualInfo>();
		Set<Integer> selectedFeatures = new HashSet<Integer>();
		
		//boot strap selected feature set with  one based on max relevancy
		FeatureMutualInfo mostRelevantFeature = getMutualInfoMaximizerScore().get(0);
		FeatureMutualInfo featureClassMutualInfo = new FeatureMutualInfo( mostRelevantFeature.getLeft(), mostRelevantFeature.getRight());
		featureJointMutualInfoList.add(featureClassMutualInfo);
		selectedFeatures.add(mostRelevantFeature.getLeft());

		//select features
		while (selectedFeatures.size() < featureClassMutualInfoList.size() ) {
			double maxScore = Double.NEGATIVE_INFINITY;
			int selectedFeature = 0;

			//all features
			for (FeatureMutualInfo featureMuInfo : featureClassMutualInfoList ) {
				int feature = featureMuInfo.getLeft();
				if (selectedFeatures.contains(feature)) {
					continue;
				}
				double sum = 0;
				for (FeaturePairMutualInfo featurePairMuInfo :  featurePairClassMutualInfoList) {
					//pair with feature already selected
					if ( featurePairMuInfo.getLeft() == feature && selectedFeatures.contains(featurePairMuInfo.getCenter()) || 
							featurePairMuInfo.getCenter() == feature && selectedFeatures.contains(featurePairMuInfo.getLeft()) ) {
						if (joinMutInfo) {
							sum +=  featurePairMuInfo.getRight();
						} else {
							FeaturePairEntropy featurePairEntropy = getFeaturePairClassEntropy(featurePairMuInfo.getLeft(), featurePairMuInfo.getCenter());
							sum +=  featurePairMuInfo.getRight() / featurePairEntropy.getRight() ;
						}
					} 
				}
					
				double score =   sum ;
				if (score > maxScore) {
					maxScore = score;
					selectedFeature = feature;
				}
			}
			//add the feature with max score
			featureClassMutualInfo = new FeatureMutualInfo( selectedFeature, maxScore);
			featureJointMutualInfoList.add(featureClassMutualInfo);
			selectedFeatures.add(selectedFeature);
		}
		return featureJointMutualInfoList;
	}
	
	/**
	 * @param featureOne
	 * @param featureTwo
	 * @return
	 */
	private FeaturePairEntropy getFeaturePairClassEntropy(int featureOne, int featureTwo) {
		FeaturePairEntropy featurePairEntropy = null;
		for (FeaturePairEntropy  thisFeaturePairEntropy : featurePairClassEntropyList) {
			if (thisFeaturePairEntropy.getLeft() == featureOne && thisFeaturePairEntropy.getCenter() == featureTwo || 
					thisFeaturePairEntropy.getLeft() == featureTwo && thisFeaturePairEntropy.getCenter() == featureOne ) {
				featurePairEntropy = thisFeaturePairEntropy;
				break;
			}
		}
		return featurePairEntropy;
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
