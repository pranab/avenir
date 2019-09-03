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

import org.apache.spark.rdd.RDD
import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.util.BasicUtils
import org.chombo.spark.common.GeneralUtility
import org.chombo.spark.common.Record
import org.chombo.math.MathUtils

/**
 * linear matrix transformation
 * @author pranab
 *
 */
object LinearMapper extends JobConfiguration with GeneralUtility {
  
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "linearMapper"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyFieldOrdinals = toIntArray(getMandatoryIntListParam(appConfig, "id.field.ordinals"))
	   val quantFieldOrdinals = toIntArray(getMandatoryIntListParam(appConfig, "quant.field.ordinals"))
	   val reatinedFieldOrdinals = toOptionalIntArray(getOptionalIntListParam(appConfig, "retained.field.ordinals"))
	   val transMatrixPath = this.getMandatoryStringParam(appConfig, "trans.matrix.path", "missing transformation matrix file path")
	   val precision = getIntParamOrElse(appConfig, "output.precision", 3)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   //mapping matrix
	   val lines  = toStringArray(BasicUtils.getFileLines(transMatrixPath))
	   val mapperData = lines.map(line => {
		  val items = BasicUtils.getTrimmedFields(line, fieldDelimIn) 
		  BasicUtils.toDoubleArray(items)
	   })
	   val mapper  = MathUtils.createMatrix(mapperData)
	   
	   val data = sparkCntxt.textFile(inputPath).cache
	   val mappedData = data.map(line => {
		   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		   val keyFields = BasicUtils.extractFieldsAsStringArray(items, keyFieldOrdinals)
		   val keyStr = BasicUtils.join(keyFields, 0, keyFields.length, fieldDelimOut)
		   
		   val quantFields = BasicUtils.extractFieldsAsDoubleArray(items, quantFieldOrdinals)
		   val data = MathUtils.createColMatrix(quantFields)
		   val mapped = MathUtils.multiplyMatrix(mapper, data)
		   val mappedArr = MathUtils.arrayFromColumnMatrix(mapped)
		   val mappedArrStr = mappedArr.map(v => BasicUtils.formatDouble(v, precision))
		   val quantStr = BasicUtils.join(mappedArrStr, 0, mappedArrStr.length, fieldDelimOut)
		   
		   var trLine = keyStr
		   reatinedFieldOrdinals match {
		     case Some(fieldOrdinals) => {
		       val retFields = BasicUtils.extractFieldsAsStringArray(items, fieldOrdinals)
		       val retStr = BasicUtils.join(retFields, 0, retFields.length, fieldDelimOut)
		       trLine += (fieldDelimOut + retStr)
		     }
		     case None =>
		   }
		   trLine += (fieldDelimOut + quantStr)
		   trLine
	   })
	   
	   if (debugOn) {
         val records = mappedData.collect
         records.slice(0, 10).foreach(r => println(r))
       }
	   
	   if(saveOutput) {	   
	     mappedData.saveAsTextFile(outputPath) 
	   }	 
	   

   }

}