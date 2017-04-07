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
import org.chombo.util.BasicUtils

object StateTransitionRate extends JobConfiguration {
  
 /**
 * @param args
 * @return
 */
   def main(args: Array[String]) {
	   val appName = "stateTransitionRate"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //add jars
	   this.addJars(sparkCntxt, appConfig, true, "lib.jars")
	   
	   //config params
	   val fieldDelimIn = appConfig.getString("field.delim.in")
	   val fieldDelimOut = appConfig.getString("field.delim.out")
	   val keyFieldOrdinals = appConfig.getIntList("key.field.ordinals").asScala.toArray
	   val timeFieldOrdinal = appConfig.getInt("time.field.ordinal")
	   val stateFieldOrdinal = appConfig.getInt("state.field.ordinal")
	   val states = appConfig.getStringList("state.values")
	   val rateTimeUnit = appConfig.getString("rate.time.unit")
	   val inputTimeUnit = appConfig.getString("input.time.unit")
	   val dateFormat : Option[SimpleDateFormat] = inputTimeUnit match {
	     case "formatted" => {
	       val inputTimeFormat = appConfig.getString("input.time.format")
	       Some(new SimpleDateFormat(inputTimeFormat))
	     }
	     case _ => None
	   }
	   val outputPrecision = appConfig.getInt("trans.rate.output.precision")
	   val debugOn = appConfig.getBoolean("debug.on")
	   val saveOutput = appConfig.getBoolean("save.output")
	  
	  //paired RDD 
	  val data = sparkCntxt.textFile(inputPath)
	  val pairedData = data.map(line => {
	    val items = line.split(fieldDelimIn)
	    
	    val keyRec = Record(items, keyFieldOrdinals)
	    //convert time stamp to epoch time
	    val dateTime = items(timeFieldOrdinal)
	    val epochTime : Long = inputTimeUnit match {
	    	case "ms" => dateTime.toLong
	    	case "sec" => dateTime.toLong * 1000
	    	case "formatted" => BasicUtils.getEpochTime(dateTime, dateFormat.get)
	    	case _ => throw new IllegalArgumentException("invalid input time unit")
	    }
	    
	    val valRec = new Record(2)
	    valRec.addLong(epochTime)
	    valRec.addString(items(stateFieldOrdinal))
	    (keyRec, valRec)
	  })	   
	   
	  //partition by key
	  if (debugOn)
		  println("size before grouping:" + pairedData.count())
	  val partDate = pairedData.combineByKey(x => List[Record](x), (c : List[Record], y) => y :: c, 
	      (c1 : List[Record], c2 : List[Record]) => c1 ::: c2)
	  if (debugOn)
		  println("size after grouping:" + partDate.count())
	  
	  
	  //state transition and time elapsed
	  val stateTrans = partDate.mapValues( d => {
	    //sort each partition by time 
	    val sotrtedAr = d.sortBy(a => {a.getLong(0)})
	    if (debugOn)
	    	println("state sequence length:" + sotrtedAr.length)
	    var prevRec : Record = new Record(1)
	    val stateAr = new Array[Record](sotrtedAr.length - 1)
	    for (i <- sotrtedAr.indices) {
	      i match {
	        case 0 => prevRec = sotrtedAr(0)
	        case _ => {
	          val curState = prevRec.getString(1)
	          val curRec = sotrtedAr(i)
	          val nexState = curRec.getString(1)
	          val timeElapsed = curRec.getLong(0) - prevRec.getLong(0)
	          val stateTrans = new Record(3)
	          stateTrans.add(curState, nexState, timeElapsed)
	          stateAr(i - 1) = stateTrans
	          prevRec = curRec
	        }
	      }
	    }
	    if (debugOn)
	    	println("state tran array length:" + stateAr.length)
	    	
	    //convert to rate matrix
	    val rateMatrix = new DoubleTable(states, states)
	    rateMatrix.setOutputPrecision(outputPrecision)
	    val duration = scala.collection.mutable.Map[String, Double]()
	    stateAr.foreach(a => {
	      //state transition
	      rateMatrix.add(a.getString(0), a.getString(1), 1.0)
	      if (false)
	    	  println("table[" + a.getString(0) + "," + a.getString(1) + "]=" + rateMatrix.get(a.getString(0), a.getString(1)))
	      
	      //state duration
	      val timeElapsed = a.getLong(2).toDouble
	      val timeElapsedScaled = rateTimeUnit match {
	        case BasicUtils.TIME_UNIT_WEEK => timeElapsed / BasicUtils.MILISEC_PER_WEEK;
	        case BasicUtils.TIME_UNIT_DAY => timeElapsed / BasicUtils.MILISEC_PER_DAY;
	        case BasicUtils.TIME_UNIT_HOUR => timeElapsed / BasicUtils.MILISEC_PER_HOUR;
	        case _ => throw new IllegalArgumentException("invalid rate time unit") 
	      } 
	      
	      duration(a.getString(0)) = duration.getOrElse(a.getString(0), 0.0) + timeElapsedScaled
	      if (debugOn)
	    	  println("duration " + a.getString(0) + "=" + duration(a.getString(0)))
	    })
	    
	    //convert to rate
	    states.asScala.foreach(s => {
	    	if (duration.contains(s)) {
	    		val scale = 1.0 / duration(s)
	    		if (debugOn) {
	    			println("scaling state: " + s + " scale: " + scale + " duration: " + duration(s))
	    			println("row: " + rateMatrix.serializeRow(s))
	    		} 
	    		rateMatrix.scaleRow(s, scale)
	    		val rowSum = rateMatrix.getRowSum(s)
	    		if (debugOn)
	    			println("rowSum: " + rowSum)
	    		val subRowSum = rowSum - rateMatrix.get(s, s)
	    		rateMatrix.set(s, s, -subRowSum)
	    		if (debugOn) {
	    			println("after scaling and substracting row: " + rateMatrix.serializeRow(s))
	    		} 
	    	}
	    })
	    
	    rateMatrix
	  })
	  
	  if (debugOn) {
		  val trans = stateTrans.collect
		  println("num of state trans rate tables:" + trans.length)
		  trans.foreach(t => {
			  println(t._1.toString())
			  val table = t._2 
			  println("row:" + table.getNumRow() + "col:" + table.getNumCol())
			  println(t._2.serializeTabular())
		  })
	  }
	   
	  if (saveOutput)
		  stateTrans.saveAsTextFile(outputPath) 
	  
   }  
}