package org.avenir.reinforce;

import java.util.HashMap;
import java.util.Map;

import org.chombo.util.CategoricalSampler;
import org.chombo.util.ConfigUtility;
import org.chombo.util.SimpleStat;

/**
 * SoftMax reinforcement learner
 * @author pranab
 *
 */
public class SoftMaxLearner extends ReinforcementLearner {
	private Map<String, SimpleStat> rewardStats = new HashMap<String, SimpleStat>();
	private Map<String, Double> expDistr = new HashMap<String, Double>();
	private double tempConstant;
	private double minTempConstant;
	private boolean rewardStatsModified;
	private CategoricalSampler sampler = new CategoricalSampler();
	private String tempRedAlgorithm;
	private static final String TEMP_RED_LINEAR = "linear";
	private static final String TEMP_RED_LOG_LINEAR = "logLinear";
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		tempConstant  = ConfigUtility.getDouble(config, "temp.constant", 100.0);
		minTempConstant  = ConfigUtility.getDouble(config, "min.temp.constant", -1.0);
	    tempRedAlgorithm = ConfigUtility.getString(config,"temp.reduction.algorithm", TEMP_RED_LINEAR );
        
        for (Action action : actions) {
        	rewardStats.put(action.getId(), new SimpleStat());
        }
 	}

	@Override
	public Action[] nextActions(int roundNum) {
		for (int i = 0; i < batchSize; ++i) {
			selActions[i] = nextAction(roundNum + i);
		}
		return selActions;
	}

	/**
	 * @param roundNum
	 * @return
	 */
	public Action nextAction(int roundNum) {
		double curProb = 0.0;
		Action action = null;
		
		//check for min trial requirement
		action = selectActionBasedOnMinTrial();

		if (null == action) {
			if (rewardStatsModified) {
				sampler.initialize();
				expDistr.clear();
				
				//all exp distributions 
				double sum = 0;
	            for (Action thisAction : actions) {
	            	double thisReward = rewardStats.get(thisAction.getId()).getMean();
	            	double distr = Math.exp(thisReward / tempConstant);
	            	expDistr.put(thisAction.getId(), distr);
	            	sum += distr;
	            }	
				
	            //prob distributions
	            for (Action thisAction : actions) {
	            	double distr = expDistr.get(thisAction.getId()) / sum;
	            	sampler.add(thisAction.getId(), distr);
	            }	
	            rewardStatsModified = false;
			}
            action = findAction(sampler.sample());
            
            //reduce constant
            long softMaxRound = totalTrialCount - minTrial;
            if (softMaxRound > 1) {
	            if (tempRedAlgorithm.equals(TEMP_RED_LINEAR)) {
	            	tempConstant /= softMaxRound;
	            } else if (tempRedAlgorithm.equals(TEMP_RED_LOG_LINEAR)) {
	            	tempConstant *= Math.log(softMaxRound) / softMaxRound;
	            }
	            
	            //apply lower bound
	            if (minTempConstant > 0 && tempConstant < minTempConstant) {
	            	tempConstant = minTempConstant;
	            }
            }            
		}
		
		++totalTrialCount;
		action.select();
		return action;
	}
	
	@Override
	public void setReward(String action, int reward) {
		rewardStats.get(action).add(reward);
		findAction(action).reward(reward);
		rewardStatsModified = true;
	}

}
