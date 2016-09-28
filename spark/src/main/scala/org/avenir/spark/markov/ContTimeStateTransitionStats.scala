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
	   idenMatrx.initialize(numStates, numStates,1.0)
	   val stateTransMult  = stateTrans.map(keySt => {
	     val trans = keySt._2
	     val maxRate = -trans.getMinDiagonalElement()
	     trans.scale(maxRate)
	     val discreetTrans = trans.add(idenMatrx)
	     val count = maxRate * timeHorizon
	     val limit = (4 + 6 * Math.sqrt(count) + count).toInt
	     
	     var matrixPowers = List[Matrix](idenMatrx)
	     matrixPowers = discreetTrans :: matrixPowers
	     for (i <- 2 to limit) {
	       val next = matrixPowers.head.dot(discreetTrans)
	       matrixPowers = next :: matrixPowers
	     }
	     
	     (keySt._1, matrixPowers)
	   })
	   
	   //collect and convert to map
	   val stateTransMultMap = scala.collection.mutable.Map[Record, List[Matrix]]()
	   stateTransMult.collect.foreach(s => {
	     stateTransMultMap += (s._1 -> s._2.reverse)
	   })
	   
	   
	   
   }
}