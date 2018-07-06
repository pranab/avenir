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
import org.chombo.spark.common.Record
import org.chombo.util.BasicUtils
import org.chombo.distance.InterRecordDistance
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer

/**
 * Feature relevance analysis by Relief algorithm
 * @param args
 * @return
 */
object FeatureRelevanceByRelief extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "featureRelevanceByRelief"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val maxNeighborCount = getMandatoryIntParam(appConfig, "max.neighbor.count", "missing neighbor count")
	   val recLen = getMandatoryIntParam(appConfig, "rec.len", "missing record length parameter")
	   val richAttrSchemaPath = getMandatoryStringParam(appConfig, "rich.attr.schema.path")
	   val genAttrSchema = BasicUtils.getGenericAttributeSchema(richAttrSchemaPath)
	   val distAttrSchemaPath = getMandatoryStringParam(appConfig, "dist.attr.schema.path")
	   val distAttrSchema = BasicUtils.getDistanceSchema(distAttrSchemaPath)
	   val distFinder = new InterRecordDistance(genAttrSchema, distAttrSchema, fieldDelimIn)
	   
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)

	   val data = sparkCntxt.textFile(inputPath).cache
	   val size = data.count
	   var fieldScore = data.flatMap(line => {	
		   val items = line.split(fieldDelimIn, -1)
		   val srcClsVal = items(0)
		   val neighborClsVal = items(1)
		   val polarity = if (srcClsVal.equals(neighborClsVal))  -1 else  1
		   val srcRec = items.slice(2, 2 + recLen)
		   var offset = 2 + recLen
		   val attrScores = ArrayBuffer[(Int,Double)]()
		   
		   //each neighbor
		   for (i <- 1 to maxNeighborCount) {
		     val neighborRec = items.slice(offset, offset + recLen)
		     distFinder.findDistance(srcRec, neighborRec)
		     val attrDist = distFinder.getAttributeDistances().asScala.toArray
		     
		     //each field
		     attrDist.foreach(r => {
		       val score = (r._1.toInt, r._2.toDouble * polarity)
		       attrScores += score
		     })
		     offset += recLen + 1
		   }
		   attrScores
	   }).reduceByKey((v1, v2) => v1 + v2)
	   
	   //scale
	   fieldScore = fieldScore.mapValues(v => v / size)
	   
	   if (debugOn) {
	     val fieldScoreCol = fieldScore.collect
	     fieldScoreCol.foreach(d => {
	       println(d)
	     })
	   }	
	   
	   if (saveOutput) {
	     fieldScore.saveAsTextFile(outputPath)
	   }
	   
   }

}