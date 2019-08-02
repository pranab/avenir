/*
 * beymani-spark: Outlier and anamoly detection 
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

package org.avenir.spark.sequence

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.util.BasicUtils
import org.chombo.spark.common.Record
import org.chombo.util.BaseAttribute
import com.typesafe.config.Config
import org.chombo.spark.common.GeneralUtility
import org.chombo.spark.common.SeasonalUtility
import org.hoidla.window.SizeBoundSymbolWindow
import org.chombo.stats.CategoricalHistogramStat

/**
 * Builds distribution of fixed length sequences
 * @author pranab
 */
object TimeDelayEmbeddingModel extends JobConfiguration   with GeneralUtility with SeasonalUtility{
   /**
   * @param args
   * @return
   */
   def main(args: Array[String]) {
	   val appName = "markovChainPredictor"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configuration params
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val precision = getIntParamOrElse(appConfig, "output.precision", 3)
	   val keyFields = toOptionalIntArray(getOptionalIntListParam(appConfig, "id.fieldOrdinals"))
	   val attrOrd = getMandatoryIntParam(appConfig, "attr.ordinal")
	   val seqFieldOrd = getMandatoryIntParam(appConfig, "seq.fieldOrd", "missing seq field ordinal")
	   val seasonalTypeFldOrd = getOptionalIntParam(appConfig, "seasonal.typeFldOrd")
	   val seasonalTypeInData = seasonalTypeFldOrd match {
		     case Some(seasonalOrd:Int) => true
		     case None => false
	   }
	   val windowSize = getIntParamOrElse(appConfig, "window.size", 3)
	   val seasonalAnalysis = getBooleanParamOrElse(appConfig, "seasonal.analysis", false)
	   val analyzerMap = creatSeasonalAnalyzerMap(this, appConfig, seasonalAnalysis, seasonalTypeInData)
	   val analyzers = creatSeasonalAnalyzerArray(this, appConfig, seasonalAnalysis, seasonalTypeInData)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig,"save.output", true)
	   val keyLen = getKeyLength(keyFields, seasonalAnalysis) 	
	   
	   //input
	   val data = sparkCntxt.textFile(inputPath)
	   	   
	   val modelData = data.map(line => {
		 val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		 val key = Record(keyLen)
		 addPrimarykeys(items, keyFields,  key)
		 addSeasonalKeys(this, appConfig,analyzerMap, analyzers, items, seasonalAnalysis, key)
	     val value = Record(2)
	     value.addLong(items(seqFieldOrd).toLong)
	     value.addString(items(attrOrd))
	   	 (key, value)
	   }).groupByKey.map(r => {	   
	     val values = r._2.toArray.sortBy(v => v.getLong(0))
	     
	     val window = new SizeBoundSymbolWindow(windowSize)
	     val histStat = new CategoricalHistogramStat()
	     values.foreach(v => {
	       window.add(v.getString(1))
	       if (window.isFull()) {
	         val data = window.getDataWindow()
	         val base = BasicUtils.join(data, ":")
	         histStat.add(base)
	       }
	     })
	     r._1.toString() + fieldDelimOut + histStat.toString()
	   })
	   
	   if (debugOn) {
         val records = modelData.collect
         records.slice(0, 100).foreach(r => println(r))
       }
	   
	   if(saveOutput) {	   
	     modelData.saveAsTextFile(outputPath) 
	   }	 
	   
	   
   }
}