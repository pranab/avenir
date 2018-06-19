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
import org.chombo.stats.NormalDistrRejectionSampler


/**
 * Converts categorical variable values to numericalUsing leave one out algorithms
 * 
 */
object CategoricalLeaveOneOutEncoding extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "categoricalLeaveOneOutEncoding"
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
	   val classPosVal = getOptionalStringParam(appConfig, "class.pos.val") 
	  
	   val regularizationFactor = getIntParamOrElse(appConfig, "regularization.factor", 10)
	   val randStdDev = getDoubleParamOrElse(appConfig, "rand.std.dev", 0.3)
	   val trainDataSet = getMandatoryBooleanParam(appConfig, "train.data.set")
	   val targetStatFilePath = getMandatoryStringParam(appConfig, "target.stat.file.path")
	   //val targetStatFileInput = getMandatoryStringParam(appConfig, "target.stat.file.input")
	   val ouputPrecision = getIntParamOrElse(appConfig, "ouput.precision", 3)
	   
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)

	   val data = sparkCntxt.textFile(inputPath).cache
	 
	   val targetStat  = 
	   if (trainDataSet) {
		   //field ordinal and field value as key, cla count and class neg count as valuess pos
		   val fieldClassVal = data.flatMap(line => {
			   val items = line.split(fieldDelimIn, -1)
			   val classVal = items(classFieldOrdinal)
			   val iClassVal = intClassVal(classVal,  classPosVal)
			   val value =  (1, iClassVal) 
			   val fcVal = catFieldOrdinals.map(i => {
			     val key = (i.toInt, items(i))
			     (key, value)
			   })
			   fcVal
		   })
		   
		   //calculate class or target  variable stat
		   val targetStat = fieldClassVal.reduceByKey((v1, v2) => {
		     ((v1._1 + v2._1), (v1._2 + v2._2))
		   })
		   targetStat
	   } else {
	     //load target variable  stat
	     val statData = sparkCntxt.textFile(targetStatFilePath)
	     val targetStat = statData.map(line => {
	    	 val items = line.split(fieldDelimIn, -1)
	    	 ((items(0).toInt, items(1)), (items(2).toInt, items(3).toInt))
	     })
	     targetStat
	   }
	   
	   val targetStatMap = targetStat.collectAsMap
	   if (trainDataSet) {
	     if (debugOn) {
	    	 targetStatMap.foreach(v => println(v._1.toString + " -> " + v._2.toString))
	     }
	     targetStat.map(r => {
	       val ar  = new Array[String](4)
	       ar(0) = r._1._1.toString
	       ar(1) = r._1._2
	       ar(2) = r._2._1.toString
	       ar(3) = r._2._2.toString
	       ar.mkString(fieldDelimOut)
	     }).saveAsTextFile(targetStatFilePath)
	   }
	   val sampler = new NormalDistrRejectionSampler(0, randStdDev, 3.0)
	   
	   //encoding
	   val encData = data.map(line => {
	     val items = line.split(fieldDelimIn, -1)
		 val classVal = items(classFieldOrdinal)
	     val iClassVal = intClassVal(classVal,  classPosVal)
	     catFieldOrdinals.foreach(i => {
	       val catVal = items(i)
	       val stat = targetStatMap.get((i, catVal))
	       val encVal =
	       if (trainDataSet) {
	         val noiseFactor = 1.0 + sampler.sample()
	         val enc = stat match {
	           case (Some(st:(Int,Int))) => ((st._2 - iClassVal).toFloat / (st._1 - 1 + regularizationFactor)) * noiseFactor
	           case None => throw new IllegalStateException("class variable stat not found")
	         }
	         enc
	       } else {
	         val enc = stat match {
	           case (Some(st:(Int,Int))) => st._2.toFloat / (st._1 + regularizationFactor)
	           case None => throw new IllegalStateException("class variable stat not found")
	         }
	         enc
	       }
	       items(i) = BasicUtils.formatDouble(encVal, ouputPrecision)
	     })
	     items.mkString(fieldDelimOut)
	   })
	   
	   if (debugOn) {
	     val endDataCol = encData.collect
	     println("showing first 10 records only")
	     endDataCol.slice(0,10).foreach(line => println(line))
	   }
	   
	   if (saveOutput) {
		   encData.saveAsTextFile(outputPath)
	   }	   
   }
   
   /**
   * @param classVal
   * @param classValCat
   * @param classPosVal
   * @return
   */   
   def intClassVal(classVal:String, classPosVal:Option[String]) : Int = {
     val iClsVal = classPosVal match {
       case Some(clsValPos:String) => if (classVal.equals(clsValPos)) 1 else -1
       case None => classVal.toInt
     }
     iClsVal
    }

}