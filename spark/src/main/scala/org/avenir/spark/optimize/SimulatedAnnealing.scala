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


package org.avenir.spark.optimize

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.spark.common.Record
import org.chombo.util.BasicUtils
import org.avenir.optimize.BasicSearchDomain

/**
 * Performs optimization with simulated annealing. Does parallel meta optimization
 * with multiple starting solution. Can do local optimization optionally
 * @param args
 * @return
 */
object SimulatedAnnealing extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "simulatedAnnealing"
	   var inputPath: Option[String] = None 
	   var outputPath : String = ""
	   var configFile : String = ""
	   args.length match {
	     	case 3 => {
	     	  inputPath = Some(args(0))
	     	  outputPath = args(1)
	     	  configFile = args(2)
	     	}
	     	case 2 => {
	     	  outputPath = args(0)
	     	  configFile = args(1)
	     	} 
	   }
	   
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val maxNumIterations = getMandatoryIntParam(appConfig, "max.num.iterations", "missing number of iterations")
	   val numOptimizers = getMandatoryIntParam(appConfig, "num.optimizers", "missing number of optimizers")
	   val maxStepSize = getMandatoryIntParam(appConfig, "max.step.size", "missing max step size")
	   val initialTemp = getMandatoryDoubleParam(appConfig, "initial.temp","missing initial temperature")
	   val coolingRate = getMandatoryDoubleParam(appConfig, "cooling.rate.value","missing cooling rate")
	   val coolingRateGeometric = getBooleanParamOrElse(appConfig, "cooling.rate.geometric", true)
	   val tempUpdateInterval = getMandatoryIntParam(appConfig, "temp.update.interval","missing temperature update interval")
	   val worseSolnAccepProb = getDoubleParamOrElse(appConfig, "worse.soln.accept.probability", 0.80)
	   val domainCallbackClass = getMandatoryStringParam(appConfig, "domain.callback.class.name", "missing domain callback class")
	   val domainCallbackConfigFile = getMandatoryStringParam(appConfig, "domain.callback.config.file", 
	       "missing domain callback config file name")
	   val mutationRetryCountLimit = getIntParamOrElse(appConfig, "mutation.retry.count.limit",  100)
	   val locallyOptimize = getBooleanParamOrElse(appConfig, "locally.optimize", true)
	   val maxNumLocalIterations = getMandatoryIntParam(appConfig, "max.num.local iterations", "missing number of local iterations")
	   val numPartitions = getIntParamOrElse(appConfig, "num.partitions",  2)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   //callback domain class
	   val domainCallback = Class.forName(domainCallbackClass).getConstructor().newInstance().asInstanceOf[BasicSearchDomain]
	   domainCallback.initTrajectoryStrategy(domainCallbackConfigFile, maxStepSize, mutationRetryCountLimit, debugOn)
	   val brDomainCallback = sparkCntxt.broadcast(domainCallback)
	   
	   //accululators
	   val costIncreaseAcum = sparkCntxt.accumulator[Double](0.0, "costIncrease")
	   val betterSolnCount = sparkCntxt.accumulator[Long](0, "betterSolnCount")
	   val bestSolnCount = sparkCntxt.accumulator[Long](0, "bestSolnCount")
	   val worseSolnCount = sparkCntxt.accumulator[Long](0, "worseSolnCount")
	   val worseSolnAcceptCount = sparkCntxt.accumulator[Long](0, "worseSolnAcceptCount")
	   
	   
	   //starting solutions are either auto generated user provided through an input file
	   val optStartSolutions = inputPath match {
	     case Some(path:String) => {
	       //initial candidate solutions provided
	       sparkCntxt.textFile(path)
	     }
	     case None => {
	       //no input, generate initial candidates
	       val optList = (for (i <- 1 to numOptimizers) yield domainCallback.createSolution()).toList
	       sparkCntxt.parallelize(optList, numPartitions)
	     }
	   }
	   
	   //global optimization
	   val bestSolutions = optStartSolutions.mapPartitions(p => {
	     //whole partition
	     val domanCallback = brDomainCallback.value.createTrajectoryStrategyClone()
	     var res = List[(String, Double)]()
	     var count = 0
	     while (p.hasNext) {
	       if (debugOn) {
	    	   println("next partition")
	       }
	       //optimizer
	       var current = p.next
	       if (debugOn) {
	         println("current:" + current)
	       }
	       domanCallback.withCurrentSolution(current)
	       var curCost = domanCallback.getSolutionCost(current)
	       var next = ""
	       var nextCost = 0.0
	       var best = current
	       var bestCost = curCost
	       var temp = initialTemp
	       var tempUpdateCounter = 0
	       for (i <- 1 to maxNumIterations) {
	         //iteration for an optimizer
	         next = domanCallback.createNeighborhoodSolution()
	         nextCost = domanCallback.getSolutionCost(next)
	         if (debugOn) {
	        	 println("iteration: " + i + " next solution: " + next + " cost: " + nextCost + 
	        	     " current solution: " + current + " cost: " + curCost)
	         }
	         if (nextCost < curCost) {
	        	 betterSolnCount += 1
	        	 
	        	 //check with best so far
	        	 if (nextCost < bestCost) {
	        		 bestSolnCount += 1
	        		 bestCost = nextCost
	        		 best = next
	        		 if (debugOn) {
	        		   println("best: " + best + " cost: " + bestCost)
	        		 }
	        	 }
	        	 
	        	 //set current to a better one found
	        	 current = next
	        	 curCost = nextCost
	        	 domanCallback.withCurrentSolution(current)
	         } else {
	            costIncreaseAcum +=  nextCost - curCost
	            worseSolnCount += 1
	            
	        	if (Math.exp((curCost - nextCost) / temp) > Math.random()) {
	        		//set current to a worse one probabilistically with higher probability at higher temp
	        		worseSolnAcceptCount += 1        		
	        		if (debugOn) {
	        		  println("accepted higher cost solution")
	        		}
	        		current = next
	        		curCost = nextCost
	        		domanCallback.withCurrentSolution(current)
	        	}
	         }
	         
	         //temp update
	         tempUpdateCounter += 1
	         if (tempUpdateCounter == tempUpdateInterval) {
	           if (coolingRateGeometric) {
	        	 temp *= coolingRate
	           } else {
	             temp -= initialTemp - i * coolingRate
	             if (temp < 0.0){
	               temp = 0
	             }
	           }
	           tempUpdateCounter = 0
	         }
	       }
	       count = count + 1
	       res ::= (best, bestCost)
	     }
	     if (debugOn) {
	    	 println("partition size: " + count)
	     }
	     res.iterator
	   }) 

	   
	   //optional local optimization 
	   val bestSolutionsFinal = locallyOptimize match {
	     case true => {
	       val localOpt = bestSolutions.mapPartitions(p => {
	    	   val domanCallback = brDomainCallback.value.createTrajectoryStrategyClone
	           var res = List[(String, Double)]()
	           while (p.hasNext) {
	        	   val rec =  p.next
	        	   val current = rec._1
	        	   val curCost = rec._2
	        	   domanCallback.withInitialSolution(current)
	        	   domanCallback.withNeighborhoodReference(false)
	        	   var next = ""
	        	   var nextCost = 0.0
	        	   var best = current
	        	   var bestCost = curCost
	        	   for (i <- 1 to maxNumLocalIterations) {
	        		   //iteration for an optimizer
	        		   next = domanCallback.createNeighborhoodSolution()
	        		   nextCost = domanCallback.getSolutionCost(next)
	        		   if (nextCost < bestCost) {
	        			   bestCost = nextCost
	        			   best = next
	        		   }
	        	   }
	        	   res ::= (best, bestCost)
	           }
	    	   res.iterator
	       })
	       localOpt
	     }
	     
	     case false => {
	       bestSolutions
	     }
	   
	   }
	   
	   //console output
	   if (debugOn) {
	     val colBestSolutions = bestSolutionsFinal.collect
	     colBestSolutions.foreach(r => println(r))
	     
	     //other counters
	     println("better solution count:" + betterSolnCount.value + " best solution count" + bestSolnCount.value)
	     println("worse solution count:" + worseSolnCount.value + " worse solution acceptance count" + worseSolnAcceptCount.value)
	     
	     //average cost increase
	     val avCostIncreae = costIncreaseAcum.value / worseSolnCount.value
	     println("average cost increase:" + BasicUtils.formatDouble(avCostIncreae, 3))
	     
	     //estimated initial temp
	     val estInitialTemp = -avCostIncreae / Math.log(worseSolnAccepProb)
	     println("esimated initial temperature:" + BasicUtils.formatDouble(estInitialTemp, 3))
	   }	   

	   //file output
	   if (saveOutput) {
	     bestSolutionsFinal.saveAsTextFile(outputPath)
	   }
	   
   }

}