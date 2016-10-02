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


package org.avenir.spark.markov

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import org.chombo.spark.common.Record
import scala.collection.JavaConverters._
import org.apache.spark.HashPartitioner
import org.chombo.util.DoubleTable
import java.text.SimpleDateFormat
import org.chombo.util.Utility
import org.chombo.util.Matrix
import org.chombo.util.PoisonDistr

 /**
  * Calculates statistics for continuous time markov chain process
 * @param args
 * @return
 */
  object ContTimeStateTransitionStats extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "contTimeStateTransitionStats"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //config params
	   val fieldDelimIn = appConfig.getString("field.delim.in")
	   val fieldDelimOut = appConfig.getString("field.delim.out")
	   val keyFieldLen = appConfig.getInt("key.field.len")
	   val states = appConfig.getStringList("state.values")
	   val numStates = states.size()
	   val timeHorizon = appConfig.getInt("time.horizon")
	   val stateTransFilePath = appConfig.getString("state.trans.file.path")
	   val transStat = appConfig.getString("state.trans.stat")
	   val targetStates = getOptionalStringListParam(appConfig, "target.states")
	   val targetStatesIndx = targetStates match {
	     case Some(lst:java.util.List[String]) => {
	       val indxs = lst.asScala.map(s => {
	         states.indexOf(s)
	       })
	       Some(indxs)
	     }
	     case None => None
	   }
	   
	   val debugOn = appConfig.getBoolean("debug.on")
	   val saveOutput = appConfig.getBoolean("save.output")
	   	   
	   //load state transition rates
	   val stateTransData = sparkCntxt.textFile(stateTransFilePath)
	   val stateTrans = stateTransData.map(line => {
		   val items = line.substring(1, line.length()-1).split(fieldDelimIn)
		   
	       val key = Record(items, 0, keyFieldLen)
		   val trans = new Matrix(numStates, numStates)
		   trans.deseralize(items, keyFieldLen)
		   (key, trans)
	   })
  
	   //normalized state transition rate and multiplications
	   val idenMatrx = new Matrix()
	   idenMatrx.initialize(numStates, numStates,0).initializeDiagonal(1)
	   val stateTransMult  = stateTrans.map(keySt => {
	     val trans = keySt._2
	     val maxRate = -trans.getMinDiagonalElement()
	     trans.scale(maxRate)
	     val discreetTrans = trans.add(idenMatrx)
	     val count = maxRate * timeHorizon
	     val limit = (4 + 6 * Math.sqrt(count) + count).toInt
	     
	     val matrixPowers = new Array[Matrix](limit+1)
	     matrixPowers(0) = idenMatrx
	     matrixPowers(1) = discreetTrans
	     for (i <- 2 to limit) {
	       val next = matrixPowers.last.dot(discreetTrans)
	       matrixPowers(i) = next
	     }
	     
	     (keySt._1, maxRate, matrixPowers)
	   })
	   
	   //collect and convert to map and broadcast
	   val stateTransMultMap = scala.collection.mutable.Map[Record, (Double, Array[Matrix])]()
	   stateTransMult.collect.foreach(s => {
	     stateTransMultMap += (s._1 -> (s._2,s._3))
	   })
	   val brStateTransMultMap = sparkCntxt.broadcast(stateTransMultMap)
	   
	   //key and initial state
	   val initState = sparkCntxt.textFile(inputPath)
	   
	   //statistic
	   val output = initState.map(line => {
		 val items = line.split(fieldDelimIn)
		 val key = Record(items, 0, keyFieldLen)
		 
		 //initial state
		 val initState = items(keyFieldLen)
		 val initStateIndx = states.indexOf(initState)
		 
		 //optional final state
		 val endStateIndx = if (items.length > keyFieldLen + 1) {
		   val endState = items(keyFieldLen + 1)
		   states.indexOf(endState)
		 } else {
		   -1
		 }
		 val (maxRate:Double, transMult:Array[Matrix]) = brStateTransMultMap.value.get(key).get
		 val count = maxRate * timeHorizon
		 val limit = (4 + 6 * Math.sqrt(count) + count).toInt
		 val poison = new PoisonDistr(count)
		 var st = Array[Double](1)
		     
	     transStat match {
		    //how long in certain state
	     	case "stateDwellTime" => {
		      val targetStateIndx = targetStatesIndx.get.head
	     	  var sumOuter:Double = 0
		      for (i <- 0 to limit) {
		       var sumInner:Double = 0
		       for (j <- 0 to i) {
		         val startTargetPr = transMult(j).get(initStateIndx, targetStateIndx)
		         val targetEndPr = endStateIndx match {
		           case indx  if indx >= 0 => transMult(i-j).get(targetStateIndx, endStateIndx)
		           case _ => 1.0
		         }
		         sumInner += (startTargetPr * targetEndPr)
		       }
		       sumOuter += (timeHorizon/(i + 1)) * sumInner * poison.next()
		      }
		      st = Array[Double](1)
		      st(0) = sumOuter
	     	}
	     	
	     	case "StateTransitionCount" => {
		      val targetStateIndxOne = targetStatesIndx.get.head
		      val targetStateIndxTwo = targetStatesIndx.get.tail.head
	     	  var sumOuter:Double = 0
		      for (i <- 0 to limit) {
		       var sumInner:Double = 0
		       for (j <- 0 to i) {
		         val startTargetPr = transMult(j).get(initStateIndx, targetStateIndxOne)
		         val targeOneTwoPr = transMult(1).get(targetStateIndxOne, targetStateIndxTwo)
		         val targetEndPr = endStateIndx match {
		           case indx  if indx >= 0 => transMult(i-j).get(targetStateIndxTwo, endStateIndx)
		           case _ => 1.0
		         }
		         sumInner += (startTargetPr * targeOneTwoPr *targetEndPr)
		       }
		       sumOuter += sumInner * poison.next()
		      }
		      st = Array[Double](1)
		      st(0) = sumOuter
	     	}
	     	
	     	//future state probability
	     	case "futureStateProb" => {
	     	  if (endStateIndx == -1) {
	     	    throw new IllegalStateException("for future state probability, end state must be defined")
	     	  }
	     	  var sum:Double = 0
	     	  for (i <- 0 to limit) {
	     	    val intiToEndPr = transMult(i).get(initStateIndx, endStateIndx)
	     	    sum += intiToEndPr * poison.next()
	     	  }
		     st = Array[Double](1)
		     st(0) = sum
	     	}
	     	
	     	case _ => {
	    	 throw new IllegalArgumentException("invalid state transition stats")
	     	}
	     }
		 (key,st)
	   })
	   
	   if(saveOutput) {	   
		   output.saveAsTextFile(outputPath) 
   	   }
	   
   }
}