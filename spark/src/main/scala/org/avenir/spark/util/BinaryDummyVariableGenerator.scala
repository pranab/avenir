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

package org.avenir.spark.util

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.spark.common.Record
import org.chombo.util.BasicUtils

/**
 * converts categorical variable to dummy binary variables
 * 
 */
object BinaryDummyVariableGenerator extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "binaryDummyVariableGenerator"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val catFieldOrdinals = getMandatoryIntListParam(appConfig, "cat.field.ordinals").asScala.toArray
	   val catValues = scala.collection.mutable.Map[Int, Array[String]]()
	   var valCount = 0
	   catFieldOrdinals.foreach(i => {
	     val colIndex = i.toInt
	     val key = "fieldUniqueValues." + i
	     val values = getMandatoryStringListParam(appConfig, key).asScala.toArray
	     catValues += (colIndex -> values)
	     valCount += values.length
	   })
	   val caseInsensitive = getBooleanParamOrElse(appConfig, "case.insensitive", false)
	   val rowSize = getMandatoryIntParam(appConfig, "row.size")
	   val newRowSize = rowSize - catFieldOrdinals.length + valCount
	   val trueVal = getMandatoryStringParam(appConfig, "true.value")
	   val falseVal = getMandatoryStringParam(appConfig, "false.value")
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   val data = sparkCntxt.textFile(inputPath)
	   val transformedData = data.map(line => {
		   val items = line.split(fieldDelimIn, -1)
		   if (items.length != rowSize) {
		     throw new IllegalStateException("invalid row size")
		   }
		   val newRec = scala.collection.mutable.ArrayBuffer[String]()
		   for (t <- items.zipWithIndex) {
			   val optValues = catValues.get(t._2)
			   optValues match {
			     case Some(values : Array[String]) => {
			       //map to binary
			       for (value <- values) {
			         val colValue = if (caseInsensitive) t._1.toLowerCase() else t._1
			         val binValue = if (value.equals(colValue)) trueVal else falseVal
			         newRec += binValue
			       }
			     }
			     case None => {
			       //as is
			       newRec += t._1
			     }
			   }
		   }
		   
		   newRec.toArray.mkString(fieldDelimOut)
	   })	 
	   
	   if (debugOn) {
	     val colData = transformedData.collect
	     colData.foreach(line => println(line))
	   }
	   
	   if (saveOutput) {
	     transformedData.saveAsTextFile(outputPath)
	   }
   }

}