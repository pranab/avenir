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

package org.avenir.spark.sequence

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import org.chombo.spark.common.Record
import scala.collection.JavaConverters._
import org.chombo.util.BasicUtils
import org.chombo.stats.HistogramStat

object EventTimeDistribution extends JobConfiguration {
  
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "eventTimeDistribution"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyFieldOrdinals = getMandatoryIntListParam(appConfig, "id.field.ordinals").asScala.toArray
	   val timeFieldOrdinal = getMandatoryIntParam(appConfig, "time.field.ordinal")
	   val timeResolution = getStringParamOrElse(appConfig, "time.resolution", "hourOfDay")
	   val hourGranularity = getOptionalIntParam(appConfig, "hour.granularity")
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   val binWidth = hourGranularity match  {
		   	case Some(gr:Int) => gr
		   	case None => 1
	   }
	   
	   val data = sparkCntxt.textFile(inputPath)
	   val pairedData = data.map(line => {
		   val items = line.split(fieldDelimIn, -1)
		   val keyRec = Record(items, keyFieldOrdinals)
		   val dateTime = items(timeFieldOrdinal).toLong
		   val timeCycle = timeResolution match {
		   	case "hourOfDay" => {
		   		var tm = dateTime % BasicUtils.MILISEC_PER_DAY
		   		tm /= BasicUtils.MILISEC_PER_HOUR
		   		tm = hourGranularity match  {
		   			case Some(gr:Int) => tm / gr
		   			case None => tm
		   		}
		   		tm
		   	}
		   	case "dayOfWeek" => {
		   		var tm = dateTime / BasicUtils.MILISEC_PER_WEEK
		   		tm /= BasicUtils.MILISEC_PER_DAY
		   		tm
		   	}
		   }
		   val valRec = new HistogramStat(binWidth)
		   valRec.add(timeCycle)
		   (keyRec, valRec)
	   })
	   
	   //merge histograms
	   val stats = pairedData.reduceByKey((h1, h2) => h1.merge(h2))
	   
	   if (debugOn) {
	     val colStats = stats.collect
	     colStats.foreach(s => {
	       println("id:" + s._1)
	       println("distr:" + s._2)
	     })
	   }
	   
	   if (saveOutput) {
	     stats.saveAsTextFile(outputPath)
	   }
   }
}