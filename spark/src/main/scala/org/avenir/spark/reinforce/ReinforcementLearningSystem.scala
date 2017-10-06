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
import org.avenir.reinforce.ReinforcementLearnerFactory
import org.avenir.reinforce.ReinforcementLearner

object ReinforcementLearningSystem extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "reinforcementLearningSystem"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   
	   val groupFieldOrdinal = getMandatoryIntParam(appConfig, "group.field.ordinal")
	   val actionFieldOrdinal = getMandatoryIntParam(appConfig, "action.field.ordinal")
	   val countFieldOrdinal = getMandatoryIntParam(appConfig, "count.field.ordinal")
	   val rewardAvgFieldOrdinal = getMandatoryIntParam(appConfig, "reward.avg.field.ordinal")
	   val rewardStdDevFieldOrdinal = getMandatoryIntParam(appConfig, "reward.stdDev.field.ordinal")
	   val actions = BasicUtils.fromListToStringArray(getMandatoryStringListParam(appConfig, "action.list"))
	   val batchSize = getMandatoryIntParam(appConfig, "batch.size")
	   
	   
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)

	   
	   //algorithm and algorithm specific configuration
	   val learnAlgo = getMandatoryStringParam(appConfig, "learning.algorithm")
	   val appAlgoConfig = appConfig.getConfig(learnAlgo)
	   val configParams = getConfig(learnAlgo, appAlgoConfig)
	   
	   val data = sparkCntxt.textFile(inputPath)
	   
	   //key by group
	   val pairedData = data.map(line => {
	     val group = line.split(fieldDelimIn, -1)(groupFieldOrdinal)
	     (group, line)
	   })

	   //add record to learner
	   val addToLeaner = (learner:ReinforcementLearner, line:String) => {
	     val items = line.split(fieldDelimIn, -1)
	     val action = items(actionFieldOrdinal)
	     val trialCount = Integer.parseInt(items(countFieldOrdinal))
	     val rewardAvg  = java.lang.Double.parseDouble(items(rewardAvgFieldOrdinal))
	     val rewardStdDev = java.lang.Double.parseDouble(items(rewardStdDevFieldOrdinal))
	     learner.setReward(action, rewardAvg, rewardStdDev, trialCount)
	     learner
	   }
	   
	   //create learner
	   val createLearner = (line:String) => {
	     val learner = ReinforcementLearnerFactory.create(learnAlgo, actions, configParams)
	     addToLeaner(learner, line)
	   }
	   
	   
	   //merge learners
	   val mergeLearners = (learnerOne : ReinforcementLearner, learnerTwo:ReinforcementLearner) => {
	     learnerOne.merge(learnerTwo)
	     learnerOne
	   }
	   
	   //build group wise learners
	   val groupedLearners =  pairedData.combineByKey(createLearner, addToLeaner, mergeLearners)
	   
	   //generate actions
	   val groupActions = groupedLearners.flatMapValues(learner => {
	     val batch = for (i <- 1 to batchSize) yield i
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
	  
	   if (saveOutput) {
	     groupActions.saveAsTextFile(outputPath)
	   }
   }
   
   /**
   * @param args
   * @return
   */
   def getConfig(learnAlgo : String, appAlgoConfig : Config) : java.util.Map[String, Object] = {
	   val configParams = new java.util.HashMap[String, Object]()
	   learnAlgo match {
	       case "randomGreedy" => {
	         configParams.put("current.decision.round", new Integer(appAlgoConfig.getInt("current.decision.round")))
	         configParams.put("random.selection.prob", new java.lang.Double(appAlgoConfig.getDouble("random.selection.prob")))
	         configParams.put("prob.reduction.algorithm", appAlgoConfig.getString("prob.reduction.algorithm"))
	         configParams.put("prob.reduction.constant", new java.lang.Double(appAlgoConfig.getDouble("prob.reduction.constant")))
	         configParams.put("auer.greedy.constant", new Integer(appAlgoConfig.getInt("auer.greedy.constant")))
	         configParams.put("decision.batch.size", new Integer(appAlgoConfig.getInt("decision.batch.size")))
	       }
	       case _ => throw new IllegalStateException("invalid RL algorithm")
	   }
	     
	   configParams
	}
  
}