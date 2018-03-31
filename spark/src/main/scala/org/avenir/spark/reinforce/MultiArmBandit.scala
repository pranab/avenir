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

package org.avenir.spark.reinforce

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import com.typesafe.config.Config
import org.chombo.util.BasicUtils
import org.avenir.reinforce.MultiArmBanditLearnerFactory
import org.avenir.reinforce.MultiArmBanditLearner

object MultiArmBandit extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "multiArmBandit"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   
	   val actions = BasicUtils.fromListToStringArray(getMandatoryStringListParam(appConfig, "action.list"))
	   val batchSize = getMandatoryIntParam(appConfig, "decision.batch.size")
	   val rewardFeedbackFilePath = getOptionalStringParam(appConfig, "reward.feedback.file.path")
	   val modelStateOutputFilePath = getMandatoryStringParam(appConfig, "model.state.output.file.path")
	   val curDecRound = getMandatoryIntParam(appConfig, "current.decision.round")
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)

	   
	   //algorithm and algorithm specific configuration
	   val learnAlgo = getMandatoryStringParam(appConfig, "learning.algorithm")
	   val appAlgoConfig = appConfig.getConfig(learnAlgo)
	   val configParams = getConfig(learnAlgo, appConfig, appAlgoConfig)
	   
	   //model state file as input
	   val data = sparkCntxt.textFile(inputPath)
	   
	   //key by group
	   val keyedData = data.map(line => {
	     val items = BasicUtils.splitOnFirstOccurence(line, fieldDelimIn, true)
	     (items(0), items(1))
	   })
	   
	   //add state record to learner
	   val addToLeaner = (learner:MultiArmBanditLearner, state:String) => {
	     learner.buildModel(state)
	     learner
	   }
	   
	   //create learner
	   val createLearner = (state:String) => {
	     val learner = MultiArmBanditLearnerFactory.create(learnAlgo, actions, configParams)
	     addToLeaner(learner, state)
	   }
	   
	   //merge learners
	   val mergeLearners = (learnerOne : MultiArmBanditLearner, learnerTwo:MultiArmBanditLearner) => {
	     learnerOne.merge(learnerTwo)
	     learnerOne
	   }

	   //build group wise learners from state
	   var groupWiseLearners =  keyedData.combineByKey(createLearner, addToLeaner, mergeLearners)

	   //optionally add reward to learners
	   groupWiseLearners = rewardFeedbackFilePath match {
	     case Some(path:String) => {
	       val rewardData = sparkCntxt.textFile(path)
	       val keyedReward = rewardData.map(line => {
	         val items = BasicUtils.splitOnFirstOccurence(line, fieldDelimIn, true)
	        (items(0), items(1))
	       })
	       
	       //update learner with reward
	       val coGroupedLearnerRewards = groupWiseLearners.cogroup(keyedReward)
	       val updatedLearners = coGroupedLearnerRewards.mapValues(v => {
	         val learner = v._1.head
	         val reawrds = v._2
	         reawrds.foreach(reward => {
	           val items = reward.split(fieldDelimIn)
	           learner.setReward(items(0), items(1).toDouble)
	         })
	         learner
	       })
	       updatedLearners
	     } 
	     
	     case None => groupWiseLearners
	   }
	   
	   //generate actions
	   val batch = for (i <- 1 to batchSize) yield i
	   val groupActions = groupWiseLearners.flatMapValues(learner => {
	     val actions = batch.map(b => {learner.nextAction().getId()})
	     actions
	   })
	    
	   if (debugOn) {
		   val actionArray = groupActions.collect
	       actionArray.foreach(a => {
	         println("group:" + a._1)
	         println("action:" + a._2)
	       })
	   }
	  
	   //save recommended actions
	   if (saveOutput) {
		   //save decisions as the primary output
		   groupActions.saveAsTextFile(outputPath)
	     
		   //model needs updating if updated with new reward or this is the first round
		   val saveModel = rewardFeedbackFilePath match {
		     case Some(path:String) => true	     
		     case None => curDecRound == 1
		   }
		   
		   //save model as the secondary output
		   if (saveModel) {
			   //save state
			   val modelState = groupWiseLearners.flatMapValues(v => {
			         val state = v.getModel()
			         state
			       }).map(kv => {
			         kv._1 + fieldDelimOut + kv._2
			       })
			    modelState.saveAsTextFile(modelStateOutputFilePath)
		   }
	   }

   }
   
   /**
   * @param args
   * @return
   */
   def getConfig(learnAlgo : String, appConfig : Config,  appAlgoConfig : Config) : java.util.Map[String, Object] = {
	   val configParams = new java.util.HashMap[String, Object]()
	   
	   //common configurations
	   val batchSize = getMandatoryIntParam(appConfig, "decision.batch.size")
	   val curDecRound = getMandatoryIntParam(appConfig, "current.decision.round")
	   val rewardScale = getIntParamOrElse(appConfig, "reward.scale", 1)
	   val minTrial = getIntParamOrElse(appConfig, "min.trial", 1)
	   configParams.put("current.decision.round", new Integer(curDecRound))
	   configParams.put("decision.batch.size", new Integer(batchSize))
	   configParams.put("reward.scale", new Integer(rewardScale))
	   configParams.put("min.trial", new Integer(minTrial))
	   
	   //algorithm specific configurations
	   learnAlgo match {
	       case MultiArmBanditLearnerFactory.RANDOM_GREEDY => {
	         configParams.put("random.selection.prob", new java.lang.Double(appAlgoConfig.getDouble("random.selection.prob")))
	         configParams.put("prob.reduction.algorithm", appAlgoConfig.getString("prob.reduction.algorithm"))
	         configParams.put("prob.reduction.constant", new java.lang.Double(appAlgoConfig.getDouble("prob.reduction.constant")))
	         configParams.put("auer.greedy.constant", new Integer(appAlgoConfig.getInt("auer.greedy.constant")))
	       }
	       case MultiArmBanditLearnerFactory.UPPER_CONFIDENCE_BOUND_ONE => {
	       }
	       case MultiArmBanditLearnerFactory.UPPER_CONFIDENCE_BOUND_TWO => {
	         configParams.put("alpha", new java.lang.Double(appAlgoConfig.getDouble("alpha")))
	       }
	       case MultiArmBanditLearnerFactory.THOMPSON_SAMPLER => {
	         configParams.put("min.sample.size", new java.lang.Integer(getIntParamOrElse(appAlgoConfig, "min.sample.size", 8)))
	         configParams.put("max.reward", new java.lang.Integer(getMandatoryIntParam(appAlgoConfig, "max.reward")))
	         configParams.put("bin.width", new java.lang.Integer(getMandatoryIntParam(appAlgoConfig, "bin.width")))
	       }
	       case MultiArmBanditLearnerFactory.OPTIMISTIC_THOMPSON_SAMPLER => {
	       }
	       case MultiArmBanditLearnerFactory.SOFT_MAX => {
	         configParams.put("temp.constant", new java.lang.Double(appAlgoConfig.getDouble("temp.constant")))
	         configParams.put("min.temp.constant", new java.lang.Double(appAlgoConfig.getDouble("min.temp.constant")))
	         configParams.put("temp.reduction.algorithm", appAlgoConfig.getString("temp.reduction.algorithm"))
	       }
	       case MultiArmBanditLearnerFactory.EXPONENTIAL_WEIGHT => {
	         configParams.put("distr.constant", new java.lang.Double(appAlgoConfig.getDouble("distr.constant")))
	       }
	       case MultiArmBanditLearnerFactory.EXPONENTIAL_WEIGHT_EXPERT => {
	         configParams.put("distr.constant", new java.lang.Double(appAlgoConfig.getDouble("distr.constant")))
	         configParams.put("num.experts", new Integer(appAlgoConfig.getInt("num.experts")))
	         configParams.put("experts",BasicUtils.intArrayFromString(appAlgoConfig.getString("experts"), 
	             BasicUtils.configDelim))
	       }
	       case MultiArmBanditLearnerFactory.INTERVAL_ESTIMATOR => {
	         configParams.put("bin.width", new java.lang.Double(appAlgoConfig.getDouble("bin.width")))
	         configParams.put("confidence.limit", new java.lang.Double(appAlgoConfig.getDouble("confidence.limit")))
	         configParams.put("min.confidence.limit", new java.lang.Double(appAlgoConfig.getDouble("min.confidence.limit")))
	         configParams.put("confidence.limit.reduction.step", new java.lang.Double(
	             appAlgoConfig.getDouble("confidence.limit.reduction.step")))
	         configParams.put("confidence.limit.reduction.round.interval", new java.lang.Double(
	             appAlgoConfig.getDouble("confidence.limit.reduction.round.interval")))
	         configParams.put("min.reward.distr.sample", new Integer(appAlgoConfig.getInt("min.reward.distr.sample")))
	       }
	       case _ => throw new IllegalStateException("invalid MAB algorithm")
	   }
	     
	   configParams
	}
  
}