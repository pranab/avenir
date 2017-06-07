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


package org.avenir.spark.similarity

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.spark.common.Record
import org.chombo.util.BasicUtils
import org.chombo.distance.InterRecordDistance
import scala.collection.mutable.ListBuffer

object GroupedRecordSimilarity extends JobConfiguration {

   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "recordSimilarity"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyFieldOrdinals = getMandatoryIntListParam(appConfig, "id.field.ordinals").asScala.toArray
	   val richAttrSchemaPath = getMandatoryStringParam(appConfig, "rich.attr.schema.path")
	   val genAttrSchema = BasicUtils.getGenericAttributeSchema(richAttrSchemaPath)
	   val distAttrSchemaPath = getMandatoryStringParam(appConfig, "dist.attr.schema.path")
	   val distAttrSchema = BasicUtils.getDistanceSchema(distAttrSchemaPath)
	   val distFinder = new InterRecordDistance(genAttrSchema, distAttrSchema, fieldDelimIn)
	   val groupFieldOrdinals = getMandatoryIntListParam(appConfig, "group.field.ordinals").asScala.toArray
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   //read input
	   val data = sparkCntxt.textFile(inputPath)
	   
	   //key with all bucket pairs and record as value
	   val groupedData = data.map(line => {
		   val items = line.split(fieldDelimIn, -1)
		   val groupRec = Record(items, groupFieldOrdinals)
		   (groupRec, line)
	   })
	   
	   //group by key and generate distances
	   val groupedDistances = groupedData.groupByKey().flatMapValues(recs => {
	     val size = recs.size
	     val records = recs.toArray
	     val distances = new ListBuffer[(Record, Record, Double)]()

	     for (i <- 0 to (size -1)) {
	       val firstKey = Record(records(i).split(fieldDelimIn, -1), keyFieldOrdinals)
	       for (j <- (i+1) to (size-1)) {
	    	   val secondKey = Record(records(j).split(fieldDelimIn, -1), keyFieldOrdinals)
	    	   val dist = distFinder.findDistance(records(i), records(j))
	    	   distances += ((firstKey, secondKey, dist))
	       }
	     }
	     distances
	   })
	   
	   if (debugOn) {
	     val distCol = groupedDistances.collect
	     distCol.foreach(d => {
	       println(d)
	     })
	   }	
	   
	   if (saveOutput) {
	     groupedDistances.saveAsTextFile(outputPath)
	   }
	   
	   
   }  
}