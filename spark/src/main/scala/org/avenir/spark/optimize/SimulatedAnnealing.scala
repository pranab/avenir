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
	   val initialTemp = getMandatoryDoubleParam(appConfig, "initial.temp","missing initial temperature")
	   val coolingRate = getMandatoryDoubleParam(appConfig, "cooling.rate","missing initial temperature")
	   val coolingRateGeometric = getBooleanParamOrElse(config, "cooling.rate.geometric", true)
	   val tempUpdateInterval = getMandatoryIntParam(appConfig, "temp.update.interval","missing temperature update interval")
	   val domainCallbackClass = getMandatoryStringParam(config, "domain.callback.class", "missing domain callback class")
	   val domainCallbackConfigFile = getMandatoryStringParam(config, "domain.callback.config.file", 
	       "missing domain callback config file name")
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   val domainCallback = Class.forName(domainCallbackClass).getConstructor().newInstance().asInstanceOf[BasicSearchDomain]
	   domainCallback.intialize(domainCallbackConfigFile)
	   val brDomainCallback = sparkCntxt.broadcast(domainCallback)
	   
	   //all optimizers
	   val optStartSolutions = inputPath match {
	     case Some(path:String) => {
	       //initial candidate solutions provided
	       sparkCntxt.textFile(path)
	     }
	     case None => {
	       //no input, generate initial candidates
	       val optList = (for (i <- 1 to numOptimizers) yield domainCallback.createSolution()).toList
	       sparkCntxt.parallelize(optList)
	     }
	   }
	   
	   val bestSolutions = optStartSolutions.mapPartitions(p => {
	     //whole partition
	     val domanCallback = brDomainCallback.value.createClone
	     var res = List[(String, Double)]()
	     while (p.hasNext) {
	       //optimizer
	       var current = p.next
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
	         if (nextCost < curCost) {
	        	 //check with best so far
	        	 if (nextCost < bestCost) {
	        		 bestCost = nextCost
	        		 best = next
	        	 }
	        	 
	        	 //set current to a better one found
	        	 current = next
	        	 curCost = nextCost
	        	 domanCallback.withCurrentSolution(current)
	         } else {
	        	if (Math.exp(curCost.toDouble - nextCost.toDouble / temp) > Math.random()) {
	        		//set current to a worse one probabilistically with hiher pribabilty at higher temp
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
	       res ::= (best, bestCost)
	     }
	     res.iterator
	   }) 

	   if (debugOn) {
	     val colBestSolutions = bestSolutions.collect
	     colBestSolutions.foreach(r => println(r))
	   }	   

	   if (saveOutput) {
	     bestSolutions.saveAsTextFile(outputPath)
	   }

	   
	   
   }

}