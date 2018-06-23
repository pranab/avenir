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

/**
 * nearest neighbors for records
 * @param args
 * @return
 */
object NearestRecords extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "nearestRecords"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val nearestNeighborByCount = getBooleanParamOrElse(appConfig, "nearestNeighborByCount", true)
	   val maxNeighborCount = getOptionalIntParam(appConfig, "max.neighbor.count")
	   val maxNeighborDist = getOptionalDoubleParam(appConfig, "max.neighbor.dist")
	   val includeFirstRecClassVal = getBooleanParamOrElse(appConfig, "include.class.val.first", true)
	   val includeSecondRecClassVal = getBooleanParamOrElse(appConfig, "include.class.val.second", true)
	   val classValOrd = getOptionalIntParam(appConfig, "class.val.ord")
	   val recLen = getMandatoryIntParam(appConfig, "rec.len", "missing record length parameter")
	   val compactOutputFormat = getBooleanParamOrElse(appConfig, "output.format.compact", true)
	   
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   var keyLen = 1
	   if (includeFirstRecClassVal) keyLen += 1
	   if (includeSecondRecClassVal) keyLen += 1
	   
	   val data = sparkCntxt.textFile(inputPath)
	   
	   //keyed by first record
	   val keyedRec = data.map(line => {
		   val items = line.split(fieldDelimIn, -1)
		   val firstRec = items.slice(0, recLen)
		   val secondRec = items.slice(recLen, 2 * recLen)
		   val dist = items(2 * recLen).toDouble
		   val keyRec = Record(keyLen)
		   if (includeFirstRecClassVal) { 
		     classValOrd match {
		       case Some(classOrd : Int) => keyRec.addString(firstRec(classOrd))
		       case None => throw new IllegalStateException("missing class field ordinal parameter")
		     }
		   }
		   if (includeSecondRecClassVal) { 
		     classValOrd match {
		       case Some(classOrd : Int) => keyRec.addString(secondRec(classOrd))
		       case None => throw new IllegalStateException("missing class field ordinal parameter")
		     }
		   }
		   keyRec.addString(firstRec.mkString(fieldDelimOut))
		   
		   val valRec = Record(2)
		   valRec.addString(secondRec.mkString(fieldDelimOut))
		   valRec.addDouble(dist)
		   (keyRec, valRec)	   
	   })
	   
	   //collect all neighbors
	   val nearestNeighbors = keyedRec.groupByKey().mapValues(r => {
	     val sortFields = Array[Int](1)
	     val neighbors = r.toArray
	     neighbors.foreach(n => n.withSortFields(sortFields))
	     scala.util.Sorting.quickSort(neighbors)
	     val nearestNeighbors = 
	     if (nearestNeighborByCount) {
	       val nearestNeighbors = maxNeighborCount match {
	         //by count
	         case Some(count : Int) => neighbors.slice(0, count)
	         case None => throw new IllegalStateException("missing max neighbor count")
	       }
	       nearestNeighbors
	     } else {
	       //by distance
	       val nearestNeighbors = maxNeighborDist match {
	         case Some(dist : Double) => neighbors.filter(r => r.getDouble(1) < dist)
	         case None => throw new IllegalStateException("missing max neighbor distance")
	       }
	       nearestNeighbors
	     }
	     nearestNeighbors
	   })
	   
	   //final output
	   val serNeighbors = 
	   if (compactOutputFormat) {
	     //all neighbors in the same record
	     val recs = nearestNeighbors.map(r => {
	       r._1.toString + fieldDelimOut + r._2.mkString(fieldDelimOut)
	     })
	     recs
	   } else {
	     //separate record for each neighbor
	     val recs = nearestNeighbors.flatMap(r => {
	       val key = r._1.toString
	       val recs = r._2.map(v => {
	         key + fieldDelimOut + v.toString
	       })
	       recs
	     })
	     recs
	   }
	   
	   if (debugOn) {
	     val serNeighborsCol = serNeighbors.collect
	     serNeighborsCol.slice(0,20).foreach(d => {
	       println(d)
	     })
	   }	
	   
	   if (saveOutput) {
	     serNeighbors.saveAsTextFile(outputPath)
	   }
	   
   }
}