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
import org.chombo.spark.common.GeneralUtility


/**
 * distance between pair of records
 * @param args
 * @return
 */
object RecordSimilarity extends JobConfiguration with GeneralUtility {
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
	   val partionFieldOrdinals = toOptionalIntArray(getOptionalIntListParam(appConfig, "part.fieldOrdinals"))
	   val keyFieldOrdinals = toIntArray(getMandatoryIntListParam(appConfig, "id.field.ordinals"))
	   val richAttrSchemaPath = getMandatoryStringParam(appConfig, "rich.attr.schema.path")
	   val genAttrSchema = BasicUtils.getGenericAttributeSchema(richAttrSchemaPath)
	   val distAttrSchemaPath = getMandatoryStringParam(appConfig, "dist.attr.schema.path")
	   val distAttrSchema = BasicUtils.getDistanceSchema(distAttrSchemaPath)
	   val distFinder = new InterRecordDistance(genAttrSchema, distAttrSchema, fieldDelimIn)
	   val numBuckets = getIntParamOrElse(appConfig, "num.buckets", 16)
	   val buckets  = List.range(0, numBuckets)
	   val interSetSimilarity = getBooleanParamOrElse(appConfig, "inter.set.similariry", false)
	   val outputKeyOnly = getBooleanParamOrElse(appConfig, "output.key.only", true)
	   val otherInputPath = 
	     if (interSetSimilarity) getMandatoryStringParam(appConfig, "other.input.path", "missing second input path") 
	     else ""
	   val precision = getIntParamOrElse(appConfig, "output.precision", 3)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   val keyLen = partionFieldOrdinals match {
	       case Some(partFields) => partFields.length + 1
	       case None => 1
	   }
	   
	   //read input
	   val data = sparkCntxt.textFile(inputPath)
	   val bucketedData = if (interSetSimilarity) { 
		   //for first set key with all bucket pairs and record as value
		   val  bucketedDataThis = data.flatMap(line => {
			   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
			   val keyRec = Record(items, keyFieldOrdinals)
			   val hash = keyRec.hashCode
			   val thisBucket = (if (hash < 0) -hash else hash) % numBuckets
			   var bucketId = 0
			   val bucketedRec = buckets.map(b => {
			     val bucketPairHash = thisBucket << 12 | b
			     val key = Record(keyLen)
			     Record.populateFieldsWithIndex(items, partionFieldOrdinals, key)
			     key.addInt(bucketPairHash)
			     val value = Record(2)
			     value.add(bucketId,line)
			     //(bucketPairHash, (bucketId,line)) 
			     (key, value)
			   })
			   
			   bucketedRec
		   }).cache
		   
		   //for second set key with all bucket pairs and record as value
		   val dataThat = sparkCntxt.textFile(otherInputPath)
		   val  bucketedDataThat = dataThat.flatMap(line => {
			   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
			   val keyRec = Record(items, keyFieldOrdinals)
			   val hash = keyRec.hashCode
			   val thisBucket = (if (hash < 0) -hash else hash) % numBuckets
			   var bucketId = 1
			   val bucketedRec = buckets.map(b => {
			     val bucketPairHash = b << 12 | thisBucket 
			     val key = Record(keyLen)
			     Record.populateFieldsWithIndex(items, partionFieldOrdinals, key)
			     key.addInt(bucketPairHash)
			     val value = Record(2)
			     value.add(bucketId,line)
			     //(bucketPairHash, (bucketId,line)) 
			     (key, value)
			   })
			   
			   bucketedRec
		   })

		   bucketedDataThis ++ bucketedDataThat
	   } else {
		   //key with all bucket pairs and record as value
		   data.flatMap(line => {
			   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
			   val keyRec = Record(items, keyFieldOrdinals)
			   val hash = keyRec.hashCode
			   val thisBucket = (if (hash < 0) -hash else hash) % numBuckets
			   var bucketId = 0
			   val bucketedRec = buckets.map(b => {
			     val bucketPairHash = if (thisBucket > b) {
			        bucketId = 0
			        thisBucket << 12 | b
			     } else { 
			        bucketId = 1
			        b << 12 | thisBucket  
			     }
			       
			     val key = Record(keyLen)
			     Record.populateFieldsWithIndex(items, partionFieldOrdinals, key)
			     key.addInt(bucketPairHash)
			     val value = Record(2)
			     value.add(bucketId,line)
			     //(bucketPairHash, (bucketId,line)) 
			     (key, value)
			   })
			   
			   bucketedRec
		   })
   	   }
	   
	   //group by key and generate distances
	   val distances = bucketedData.groupByKey().flatMapValues(recs => {
	     val firstBucket = recs.filter(r => r.getInt(0) == 0)
	     val secondBucket = recs.filter(r => r.getInt(0) == 1)
	     val distances = new ListBuffer[(Record, Record, Double)]()
	     
	     //first bucket
	     firstBucket.foreach(f => {
	       val firstRecStr = f.getString(1)
	       val firstRecAr = firstRecStr.split(fieldDelimIn, -1)
	       val firstKey = Record(firstRecAr, keyFieldOrdinals)
	       //val fistRec = if (outputKeyOnly) None else Some(Record(firstRecAr))
	       
	       //second bucket
	       secondBucket.foreach(s => {
	    	   val secondRecStr = s.getString(1)
	    	   val secondRecAr = secondRecStr.split(fieldDelimIn, -1)
	    	   val secondKey = Record(secondRecAr, keyFieldOrdinals)
	    	   val dist = if (firstKey.equals(secondKey)) 0 else  distFinder.findDistance(firstRecStr, secondRecStr)
	    	   distances +=  (if (outputKeyOnly) {
	    		 (firstKey, secondKey, dist)
	    	   } else {
	    	     (Record(firstRecStr), Record(secondRecStr), dist)
	    	   })
	       })
	     })
	     distances
	   }).map(r => {
	     val key = r._1
	     val value = r._2
	     val partStr = partionFieldOrdinals match {
	       case Some(partFields) => key.toString(0, key.size -1)
	       case None => ""
	     }
	     partStr + value._1.toString() + value._2.toString() + BasicUtils.formatDouble(value._3, precision)
	   })
	   
	   if (debugOn) {
	     val distCol = distances.collect
	     distCol.foreach(d => println(d))
	   }	
	   
	   if (saveOutput) {
	     distances.saveAsTextFile(outputPath)
	   }
	   
   }
}