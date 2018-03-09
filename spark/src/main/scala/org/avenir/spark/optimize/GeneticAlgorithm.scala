
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
import org.chombo.util.BasicUtils
import org.avenir.optimize.BasicSearchDomain
import org.avenir.optimize.SolutionPopulation
import org.avenir.optimize.SolutionWithCost
import org.avenir.optimize.PopulationSearchDomain

object GeneticAlgorithm extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "geneticAlgorithm"
	   val Array(outputPath: String, configFile: String) = getCommandLineArgs(args, 2)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val numGenerations = getMandatoryIntParam(appConfig, "num.generations", "missing number of generations")
	   val populationSize = getMandatoryIntParam(appConfig, "population.size", "missing population size")
	   val numOptimizers = getMandatoryIntParam(appConfig, "num.optimizers", "missing number of optimizers")
	   val crossOverProb = getDoubleParamOrElse(appConfig, "cross.over.probability", 0.98)
	   val mutationProb = getMandatoryDoubleParam(appConfig, "mutation.probability","missing mutation.probability")
	   val mutationRetryCountLimit = getMandatoryIntParam(appConfig, "mutation.retry.count.limit",  "missing mutation retry count limit")
	   val crossOverRetryCountLimit = getMandatoryIntParam(appConfig, "cross.over.retry.count.limit",  "missing cross over retry count limit")

	   val domainCallbackClass = getMandatoryStringParam(appConfig, "domain.callback.class.name", "missing domain callback class")
	   val domainCallbackConfigFile = getMandatoryStringParam(appConfig, "domain.callback.config.file", 
	       "missing domain callback config file name")
	   val numPartitions = getIntParamOrElse(appConfig, "num.partitions",  2)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   //callback domain class
	   val domainCallback = Class.forName(domainCallbackClass).getConstructor().newInstance().
	       asInstanceOf[PopulationSearchDomain]
	   domainCallback.intitPopulationStrategy(domainCallbackConfigFile, crossOverRetryCountLimit, 
	       mutationRetryCountLimit, debugOn)
	   val brDomainCallback = sparkCntxt.broadcast(domainCallback)
	   
	   val optList = (for (i <- 1 to numOptimizers) yield i).toList
	   val optimizers = sparkCntxt.parallelize(optList, numPartitions)
	   
	   val bestSolutions = optimizers.mapPartitions(p => {
	     //whole partition
	     val domanCallback = brDomainCallback.value.createPopulationStrategyClone
	     var population = new SolutionPopulation()
	     val selPopulation = new SolutionPopulation()
	     var childPopulation = new SolutionPopulation()
	     var res = List[SolutionWithCost]()
	     var count = 0
	     
	     //all items in partition
	     while (p.hasNext) {
	       if (debugOn) {
	    	   println("next partition")
	       }
	       var current = p.next
	       population.initialize()
	       selPopulation.initialize()
	       childPopulation.initialize()
	       
	       //initial population
	       for (i <- 1 to populationSize) {
	    	   val soln = domanCallback.createSolution()
	    	   val cost = domanCallback.getSolutionCost(soln)
	    	   population.add(soln,cost)
	       }
	       var best:SolutionWithCost = null
	       
	       //all generations
	       for (i <- 1 to numGenerations) {
	    	   //selected population
	    	   selectPopulation(population, selPopulation, populationSize)
	    	   
	    	   //reproduce
	    	   for (i <- 0 to populationSize-1) {
	    	     //parents
	    	     val parents = (i % 2) match  {
	    	       case 0 => (selPopulation.getSolution(i), selPopulation.getSolution(i+1))
	    	       case 1 => (selPopulation.getSolution(i), selPopulation.getSolution(i-1))
	    	     }
	    	     
	    	     //create child
	    	     val child = reproduce(parents, domanCallback,crossOverProb, mutationProb)
	    	     val childCost = domanCallback.getSolutionCost(child)
	    	     childPopulation.add(child,childCost)
	    	   }
	    	   
	    	   var cildBest = childPopulation.findBest()
	    	   best = i match {
	    	     case 1 => cildBest
	    	     case _ => if (cildBest.getCost() < best.getCost()) cildBest else best
	    	   }
	    	   population = childPopulation
	    	   childPopulation = new SolutionPopulation()
	       }
	       res ::= best
	     }
	     res.iterator
	   })
	   
	   //console output
	   if (debugOn) {
	     val colBestSolutions = bestSolutions.collect
	     colBestSolutions.foreach(r => println(r))
	   }	  
	   
	   //file output
	   if (saveOutput) {
	     bestSolutions.saveAsTextFile(outputPath)
	}
	   
	   
   }
   
   /**
   * @param population
   * @param selPopulation
   * @param populationSize
   */
   def selectPopulation(population:SolutionPopulation, selPopulation:SolutionPopulation, populationSize:Int)  {
	   selPopulation.initialize()
	   for (i <- 1 to populationSize) {
	     val sel = population.binaryTournament()
	     selPopulation.add(sel)
	   }	
   }
   
   /**
   * @param parents
   * @param domanCallback
   * @param crossOverProb
   * @param mutationProb
   * @return
   */
   def reproduce(parents: (SolutionWithCost, SolutionWithCost), domanCallback: PopulationSearchDomain,
       crossOverProb:Double, mutationProb:Double) : String = {
	 //cross over
     var child =  (Math.random() < crossOverProb) match {
       case true => domanCallback.crossOverForOne(parents._1.getSolution(), parents._2.getSolution()) 
       case false => parents._1.getSolution()
     }
     
     //mutate
     child = (Math.random() < mutationProb) match {
       case true => domanCallback.mutateSolution(child)
       case false => child
     }
     child
   }
}