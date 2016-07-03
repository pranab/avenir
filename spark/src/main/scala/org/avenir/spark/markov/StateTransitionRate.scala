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

object StateTransitionRate extends JobConfiguration {
  
 /**
 * @param args
 * @return
 */
   def main(args: Array[String]) {
	   val Array(master: String, inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf("stateTransitionRate", config)
	   val sparkCntxt = new SparkContext(sparkConf)
	   
	   val fieldDelimIn = config.getString("app.field.delim.in")
	   val fieldDelimOut = config.getString("app.field.delim.out")
	   val keyFieldOrdinals = config.getIntList("app.key.field.ordinals").asScala
	   val timeFieldOrdinal = config.getInt("app.time.field.ordinal")
	   val stateFieldOrdinal = config.getInt("app.state.field.ordinal")
	   
	  
	  //paired RDD 
	  val data = sparkCntxt.textFile(inputPath)
	  val pairedData = data.map(line => {
	    val items = line.split(fieldDelimIn)
	    
	    val keyRec = new Record(keyFieldOrdinals.length)
	    keyFieldOrdinals.foreach(ord => {
	      keyRec.addString(items(ord))
	    })
	    
	    val valRec = new Record(2)
	    valRec.addLong(items(timeFieldOrdinal))
	    valRec.addString(items(stateFieldOrdinal))
	    (keyRec, valRec)
	  })	   
	   
	  //partition by key
	  val partDate = pairedData.groupByKey(4)
	  
	  //state transition and time elapsed
	  val stateTrans = partDate.mapValues( d => {
	    //sort each partition by time 
	    val ar = d.toArray
	    val sotrtedAr = ar.sortBy(a => {a.getLong(0)})
	    
	    var prevRec : Record = new Record(1)
	    val stateAr = new Array[Record](sotrtedAr.length - 1)
	    for (i <- sotrtedAr.indices) {
	      i match {
	        case 0 => prevRec = sotrtedAr(0)
	        case _ => {
	          val curState = prevRec.getString(1)
	          val nexState = sotrtedAr(i).getString(1)
	          val timeElapsed = sotrtedAr(i).getLong(0) - prevRec.getLong(0)
	          val stateTrans = new Record(3)
	          stateTrans.add(curState, nexState, timeElapsed)
	          stateAr(i - 1) = stateTrans
	        }
	        
	      }
	      
	    }
	    stateAr
	  })
	  
	  
	  
   }  
}