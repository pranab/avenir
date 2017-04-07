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
import scala.collection.JavaConverters._
import org.chombo.spark.common.Record
import org.chombo.util.BasicUtils
import org.chombo.spark.common.RecordPartitioner

/**
 * @param args
 * @return
 */
object SequenceGenerator extends JobConfiguration {

   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "sequenceGenerator"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyFieldOrdinals = getMandatoryIntListParam(appConfig, "id.field.ordinals").asScala.toArray
	   val valFieldOrdinals = getMandatoryIntListParam(appConfig, "val.field.ordinals").asScala.toArray
	   val seqField = Array[Int](1)
	   seqField(0) = getMandatoryIntParam(appConfig, "seq.field")
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)

	   //read input
	   val data = sparkCntxt.textFile(inputPath)

	   //key value records
	   val keyedData = data.map(line => {
		   val items = line.split(fieldDelimIn, -1)
		   val keyRec = Record(items, keyFieldOrdinals)
		   val valRec = Record(items, valFieldOrdinals).withSortFields(seqField)
		   (keyRec, valRec)
	   })
	   
	   //sort values
	   val sortedData = keyedData.groupByKey.mapValues(vals => vals.toList.sorted)
	   
	   if (debugOn) {
	     val distCol = sortedData.collect
	     distCol.foreach(d => {
	       println(d)
	     })
	   }	
	   
	   if (saveOutput) {
	     sortedData.saveAsTextFile(outputPath)
	   }
	   

  }
}