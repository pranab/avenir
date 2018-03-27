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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import scala.collection.JavaConverters._
import com.typesafe.config.Config
import org.chombo.spark.common.Record
import org.chombo.util.BasicUtils
import org.avenir.optimize.BasicSearchDomain

/**
 * Performs optimization with random search. It's the simplest possible
 * solution. Can do local optimization optionally
 * @param args
 * @return
 */
object RandomSearch extends JobConfiguration {
   /**
    * @param args
    * @return
    */
    def main(args: Array[String]) {
	   val appName = "randomSearch"
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
	   val domainCallbackClass = getMandatoryStringParam(appConfig, "domain.callback.class.name", "missing domain callback class")
	   val domainCallbackConfigFile = getMandatoryStringParam(appConfig, "domain.callback.config.file", 
	       "missing domain callback config file name")
	   val numPartitions = getIntParamOrElse(appConfig, "num.partitions",  2)
	   val locallyOptimize = getBooleanParamOrElse(appConfig, "locally.optimize", true)
	   val localSearchStrategy = getConditionalMandatoryStringParam(locallyOptimize, appConfig, "local.search.strategy", 
	       "missing local search strategy")
	   val maxNumLocalIterations = getConditionalMandatoryIntParam(locallyOptimize, appConfig, "max.num.local iterations", 
	       "missing number of local iterations")
	   val localSolnOutputFile = getConditionalMandatoryStringParam(locallyOptimize, appConfig, "local.soln.file.name", 
	       "missing local solution output file name")
	   val maxStepSize = getMandatoryIntParam(appConfig, "max.step.size", "missing max step size")
	   val mutationRetryCountLimit = 1
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)

	   //callback domain class
	   val domainCallback = Class.forName(domainCallbackClass).getConstructor().newInstance().asInstanceOf[BasicSearchDomain]
	   domainCallback.initTrajectoryStrategy(domainCallbackConfigFile, maxStepSize, mutationRetryCountLimit, debugOn)
	   val brDomainCallback = sparkCntxt.broadcast(domainCallback)
	   
	   //starting solutions are either auto generated user provided through an input file
	   val optStartSolutions = inputPath match {
	     case Some(path:String) => {
	       //initial candidate solutions provided
	       sparkCntxt.textFile(path, numPartitions)
	     }
	     case None => {
	       //no input, generate initial candidates
	       val optList = (for (i <- 1 to numOptimizers) yield domainCallback.createSolution()).toList
	       sparkCntxt.parallelize(optList, numPartitions)
	     }
	   }
	   
	   //global optimization
	   val globSolutions = optStartSolutions.mapPartitions(p => {
	     //whole partition
	     val domanCallback = brDomainCallback.value.createTrajectoryStrategyClone()
	     
	     val solnCosts = p.map(soln => {
	        val cost = domanCallback.getSolutionCost(soln)
	        (soln, cost)
	      })
	      solnCosts
	   })
	   
	   //ascending sort by cost
	   val sortedGlobalSoutions = globSolutions.sortBy(s => s._2, true)
	   val colSortedSoutions = sortedGlobalSoutions.collect
	   
	   //console output
	   if (debugOn) {
	     println("global solutions")
	     colSortedSoutions.foreach(r => println(r))
	   }
	   
	   //save to file
	   if (saveOutput) {
		   sortedGlobalSoutions.saveAsTextFile(outputPath)
	   }
	   
	   //local search around globally best
	   val globalBestSolution = colSortedSoutions(0)
	   if (locallyOptimize) {
	     val domanCallbackClone = domainCallback.createTrajectoryStrategyClone()

	     //ascending sort by cost
	     val sortedLocalSoutions = localFocussedSearch(globalBestSolution._1, domanCallbackClone, 
	         maxNumLocalIterations,numPartitions, brDomainCallback,  sparkCntxt)
	     val colSortedSoutions = sortedLocalSoutions.collect
	     
	     //if locally best solution is better than global best
	     val localBestSolution = colSortedSoutions(0) 
	     if (localBestSolution._2 < globalBestSolution._2) {
	       if (debugOn) {
	    	   println("local solutions")
	    	   colSortedSoutions.foreach(r => println(r))
	       }
	       
	       if (saveOutput) {
	    	   sortedGlobalSoutions.saveAsTextFile(localSolnOutputFile)
	       }
	     }
	   } 
	  
    }
   
    /**
	 * @param initialSolution
	 * @param domanCallback
	 * @param maxNumLocalIterations
	 * @param numPartitions
	 * @param brDomainCallback
	 * @param sparkCntxt
	 * @return
    */
    def localFocussedSearch(initialSolution:String, domanCallback : BasicSearchDomain, maxNumLocalIterations : Int,
       numPartitions:Int, brDomainCallback:Broadcast[BasicSearchDomain],  sparkCntxt:SparkContext) : 
       RDD[(String, Double)] = {
	     domanCallback.withInitialSolution(initialSolution)
	     domanCallback.withNeighborhoodReference(false)

	     //candidate local solutions
	     val localCandidateSolutions = (for (i <- 1 to maxNumLocalIterations) yield 
	       domanCallback.createNeighborhoodSolution()).toList
	     val optStartSolutions = sparkCntxt.parallelize(localCandidateSolutions, numPartitions)
     
	     //find costs
	     val localSolutions = optStartSolutions.mapPartitions(p => {
	    	 //whole partition
	    	 val domanCallback = brDomainCallback.value.createTrajectoryStrategyClone()
	     
	    	 val solnCosts = p.map(soln => {
	    		 val cost = domanCallback.getSolutionCost(soln)
	    		 (soln, cost)
	    	 })
	         solnCosts
	     })
	     
	     //ascending sort by cost
	     val sortedLocalSoutions = localSolutions.sortBy(s => s._2, true)
	     return sortedLocalSoutions
    }
    
    
   
}