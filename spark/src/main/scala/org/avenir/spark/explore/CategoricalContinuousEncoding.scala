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

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.util.BasicUtils

/**
 * Converts categorical variable values to numerical. Supports two algorithms : supervised ration
 * weight of evidence
 * 
 */
object CategoricalContinuousEncoding extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "categoricalContinuousEncoding"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val catFieldOrdinals = getMandatoryIntListParam(appConfig, "cat.field.ordinals").asScala.toArray
	   val classFieldOrdinal = getMandatoryIntParam(appConfig, "class.field.ordinal")
	   val classPosVal = getMandatoryStringParam(appConfig, "class.pos.val")
	   val encodingStrategy = getStringParamOrElse(appConfig, "encoding.strategy", "supervisedRatio")
	   val isWeightOfEvidence = encodingStrategy.equals("weightOfEvidence")
	   val scale = getIntParamOrElse(appConfig, "scale", 100)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   val data = sparkCntxt.textFile(inputPath)
	 
	   //field ordinal and field value as key, cla count and class neg count as valuess pos
	   val fieldClassVal = data.flatMap(line => {
		   val items = line.split(fieldDelimIn, -1)
		   val classVal = items(classFieldOrdinal)
		   val value = if(classVal.equals(classPosVal)) (1, 0) else (0, 1)
		   val fcVal = catFieldOrdinals.map(i => {
		     val key = (i.toInt, items(i))
		     (key, value)
		   })
		   if (isWeightOfEvidence) {
			   val allKey = (-1, "*")
			   fcVal ++ Array((allKey, value))
		   } else {
		     fcVal
		   }
	   })
	   
	   //accumulate counts
	   var fieldClassValArr = fieldClassVal.reduceByKey((v1, v2) => {
	     ((v1._1 + v2._1), (v1._2 + v2._2))
	   }).collect
	   
	   //encoded values
	   val encValues = if (isWeightOfEvidence) {
		   val allCount = fieldClassValArr.filter(v => v._1._1 == -1)(0)
		   fieldClassValArr = fieldClassValArr.filter(v => v._1._1 >= 0)
		   val encValues = fieldClassValArr.map(v => {
		     var woe = v._2._1.toDouble / allCount._2._1
		     val negCount = if (v._2._1 == 0) 1 else v._2._1
		     woe /= negCount.toDouble / allCount._2._2
		     woe = Math.log(woe)
		     woe *= scale
		     (v._1._1, v._1._2, woe.toInt)
		   })
		   encValues
	   } else {
	     val encValues = fieldClassValArr.map(v => {
	       val sr = (v._2._1 * scale) /  (v._2._1 + v._2._2)
	       (v._1._1, v._1._2, sr)
	     })
	     encValues
	   }
	   
	   if (debugOn) {
	     encValues.foreach(line => println(line))
	   }
	   
	   if (saveOutput) {
		   val encValuesRdd = sparkCntxt.parallelize(encValues, 1)
		   encValuesRdd.saveAsTextFile(outputPath)
	   }
   }
}