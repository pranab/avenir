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
	   val transStats = appConfig.getString("state.trans.stats")
	   val targetState = appConfig.getString("target.state")
	   val targetStateIndx = states.indexOf(targetState)
	   val endStateDefined = appConfig.hasPath("end.state")
	   val endStateIndx = endStateDefined match  {
	     case true => states.indexOf(appConfig.getString("end.state"))
	     case false => -1
	   }
	   val debugOn = appConfig.getBoolean("debug.on")
	   val saveOutput = appConfig.getBoolean("save.output")
	   	   
	   //state transition rates
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
	   
	   //collect and convert to map
	   val stateTransMultMap = scala.collection.mutable.Map[Record, (Double, Array[Matrix])]()
	   stateTransMult.collect.foreach(s => {
	     stateTransMultMap += (s._1 -> (s._2,s._3))
	   })
	   
	   //initial state
	   val brStateTransMultMap = sparkCntxt.broadcast(stateTransMultMap)
	   val initState = sparkCntxt.textFile(inputPath)
	   
	   //statistic
	   val stat = transStats match {
	     case "dwellTime" => {
		   initState.map(line => {
		     val items = line.split(fieldDelimIn)
		     val key = Record(items, 0, keyFieldLen)
		     val initState = items(keyFieldLen)
		     val initStateIndx = states.indexOf(initState)
		     val (maxRate:Double, transMult:Array[Matrix]) = brStateTransMultMap.value.get(key).get
		     val count = maxRate * timeHorizon
		     val limit = (4 + 6 * Math.sqrt(count) + count).toInt
		
		     val poison = new PoisonDistr(count)
		     
		     var sumOuter:Double = 0
		     for (i <- 0 to limit) {
		       var sumInner:Double = 0
		       for (j <- 0 to i) {
		         val startTargetPr = transMult(j).get(initStateIndx, targetStateIndx)
		         val targetEndPr = endStateDefined match {
		           case true => transMult(i-j).get(targetStateIndx, endStateIndx)
		           case false => 1.0
		         }
		         sumInner += (startTargetPr * targetEndPr)
		       }
		       sumOuter += (timeHorizon/(i + 1)) * sumInner * poison.next()
		     }
		     (key,sumOuter)
		   })

	     }
	     
	     case _ => {
	    	 throw new IllegalArgumentException("invalid state transition stats")
	     }
	   }
	   
	   if(saveOutput) {	   
	   	stat.saveAsTextFile(outputPath) 
   	   }
	   
   }
}