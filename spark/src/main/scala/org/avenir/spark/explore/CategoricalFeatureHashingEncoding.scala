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

/**
 * Converts high cardinality categorical variable values to numerical using feature hashing
 * 
 */
object CategoricalFeatureHashingEncoding extends JobConfiguration with GeneralUtility {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "categoricalFeatureHashingEncoding"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val catFieldOrdinals = toIntArray(getMandatoryIntListParam(appConfig, "cat.fieldOrdinals"))
	   val encodingVecSize = getMandatoryIntParam(appConfig, "encoding.size", "missing encoding vector size")
	   val indexHashingAlgo = this.getStringParamOrElse(appConfig, "index.hashingAlgo", "default")
	   val signHashingAlgo = this.getStringParamOrElse(appConfig, "sign.hashingAlgo", "FNV")
	   val rowSize = getMandatoryIntParam(appConfig, "row.size")
	   val newRowSize = rowSize - catFieldOrdinals.length + encodingVecSize
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)

	   val data = sparkCntxt.textFile(inputPath).cache
	   val transformedData = data.map(line => {
		   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		   if (items.length != rowSize) {
		     throw new IllegalStateException("invalid row size")
		   }
		   val catFields = BasicUtils.extractFieldsAsStringArray(items, catFieldOrdinals)
		   val otherFields = BasicUtils.extractRemainingFieldsAsStringArray(items, catFieldOrdinals)
		   val featureHash = getFeatureHash(catFields, encodingVecSize, indexHashingAlgo, signHashingAlgo)
		   require(catFields.length + featureHash.length == newRowSize, "new row is not expected size")
		   otherFields.mkString(fieldDelimOut) + fieldDelimOut + BasicUtils.join(featureHash, fieldDelimOut)
		   
	   })
	   
	   if (debugOn) {
         val records = transformedData.collect
         records.slice(0, 100).foreach(r => println(r))
       }
	   
	   if(saveOutput) {	   
	     transformedData.saveAsTextFile(outputPath) 
	   }	 
   }

   /*
	* @param appName
	* @param config
	* @param includeAppConfig
	* @return
	*/
   def getFeatureHash(catFields: Array[String], encodingVecSize:Int, indexHashingAlgo:String, signHashingAlgo:String) : 
	 Array[Int] =  {
     val hashed = Array.fill[Int](encodingVecSize)(0)
     catFields.foreach(v => {
       val indxHash = BasicUtils.hashCode(v, indexHashingAlgo) % encodingVecSize
       val signHash = BasicUtils.hashCode(v, signHashingAlgo) * 2
       if (signHash == 1)
         hashed(indxHash) += 1
       else 
         hashed(indxHash) -= 1
     })
     hashed
   }
   
}