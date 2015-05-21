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
	private double tempConstant;
	private CategoricalSampler sampler = new CategoricalSampler();
	private String tempRedAlgorithm;
	private static final String TEMP_RED_LINEAR = "linear";
	private static final String TEMP_RED_LOG_LINEAR = "logLinear";
	
	@Override
	public void initialize(Map<String, Object> config) {
		super.initialize(config);
		tempConstant  = ConfigUtility.getDouble(config, "temp.constant", 100.0);
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
			sampler.initialize();
            for (Action thisAction : actions) {
            	double thisReward = rewardStats.get(thisAction.getId()).getMean();
            	double distr = Math.exp(thisReward / tempConstant);
            	sampler.add(thisAction.getId(), distr);
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
	}

}
