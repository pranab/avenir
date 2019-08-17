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
package org.avenir.spark.explore

import org.apache.spark.rdd.RDD
import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.util.BasicUtils
import org.chombo.spark.common.GeneralUtility
import org.chombo.spark.common.Record
import org.chombo.stats.HistogramStat


/**
 * Finds deviation between 2 distributions based on KolmogorovSmirnov test
 * 
 */
object KolmogorovSmirnovModelDrift extends JobConfiguration with GeneralUtility {
  
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "kolmogorovSmirnovModelDrift"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyLen = getMandatoryIntParam(appConfig, "key.length", "missing key length")
	   val sigLevel = this.getDoubleParamOrElse(appConfig, "significance.level", .05)
	   val c = Math.sqrt(-0.5 * Math.log(sigLevel))
	   val precision = getIntParamOrElse(appConfig, "output.precision", 3)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)

	   val data = sparkCntxt.textFile(inputPath).cache
	   val devData = data.map(line => {
		   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		   val key = Record(items, 0, keyLen)
		   (key, line)
	   }).groupByKey.mapValues(v => {
	     val histData = v.toArray
	     if (histData.length == 2) {
	    	 val histOne = HistogramStat.createHistogram(histData(0), keyLen)
	    	 val histTwo = HistogramStat.createHistogram(histData(1), keyLen)
	    	 val ksStat = histOne.getKolmogorovSmirnovStatistic(histTwo)
	    	 val countOne = histOne.getCount().toDouble
	    	 val countTwo = histTwo.getCount().toDouble
	    	 val critVal = c * Math.sqrt((countOne + countTwo) / (countOne * countTwo))
	    	 val deviated = ksStat > critVal
	    	 (ksStat, deviated, true)
	     } else {
	         (0.0, false, false)
	     }
	   }).filter(r => r._2._3).map(r => {
	     r._1.toString + fieldDelimOut + BasicUtils.formatDouble(r._2._1, precision)  + fieldDelimOut + r._2._2
	   })
	   
	   if (debugOn) {
         val records = devData.collect
         records.slice(0, 10).foreach(r => println(r))
       }
	   
	   if(saveOutput) {	   
	     devData.saveAsTextFile(outputPath) 
	   }	 
	   
   }
}