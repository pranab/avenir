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

package org.avenir.spark.association

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import com.typesafe.config.Config
import org.apache.spark.rdd.RDD
import org.chombo.util.BasicUtils
import org.chombo.spark.common.Record
import scala.collection.mutable.ArrayBuffer
import org.chombo.spark.common.GeneralUtility

/**
* @param keyFields
* @return
*/
object FrequentItemsApriori extends JobConfiguration with GeneralUtility {
  
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "contTimeStateTransitionStats"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //config params
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val useTransId = getBooleanParamOrElse(appConfig, "use.transId", true)
	   val tranIdOrd = getOptionalIntParam(appConfig, "tran.idOrd")
	   val withTransId = tranIdOrd match {
	       case Some(ord) => true
		   case None => false
	   }
	   val keyFields = getOptionalIntListParam(appConfig, "id.fieldOrdinals")
	   val keyFieldOrdinals = toOptionalIntArray(keyFields)
	   val maxItemSetLength = getOptionalIntParam(appConfig, "max.itemSetLength")
	   val minSupport = getMandatoryDoubleParam(appConfig, "min.support", "missing minimumm support")
	   val itemFiledOrdinals = toIntList(getMandatoryIntListParam(appConfig, "item.filedOrdinals", "missing items field ordinals"))
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   val baseKeyLen = getOptinalArrayLength(keyFieldOrdinals)
	   
	   //input
	   val data = sparkCntxt.textFile(inputPath)
	   val counts = data.map(line => {
		   val fields = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		   val key = Record(baseKeyLen)
		   populateFields(fields, keyFieldOrdinals, key)
	       (key, 1)
	   }).reduceByKey((v1, v2) => v1 + v2).collectAsMap

	   var done = false
	   var itemSetLen = 1
	   val itemsSets : Option[RDD[(Record, Record)]] = None
	   
	   while(!done) {
	     val data = sparkCntxt.textFile(inputPath)
	     val keyLen = baseKeyLen + itemSetLen
	     
	     itemsSets match {
	       //items sets exits
	       case Some(itemSets) => {
	         itemSets.map(r => {
	           val key = Record(r._1, 0, baseKeyLen)
	           
	         })
	         val cartJoined = itemSets.cartesian(data)
	       }
	        
	       //first time
	       case None => {
		     val itemSets = data.flatMap(line => {
		    	 val fields = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		         itemFiledOrdinals.map(i => {
		    	   val key = Record(keyLen)
		    	   populateFields(fields, keyFieldOrdinals, key)
		    	   key.addString(fields(i))
		          
		    	   val value = tranIdOrd match {
		    	     case Some(ord) => Record(fields(ord))
		    	     case None => Record(1, 1)
		    	   }
		    	   (key, value)
		         })
		     }).reduceByKey((v1, v2) => if (withTransId) Record(v1, v2) else Record(1, v1.getInt(0) + v1.getInt(0)))
		     
	       }
	     }
	     
	   }

   }

}