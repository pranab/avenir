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
import org.chombo.math.MathUtils
import org.avenir.util.PrincipalCompState

/**
* Online PCA with Spirit algorithm
* 
*/
object IncrementalPrincipalComponent extends JobConfiguration with GeneralUtility {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "incrementalPrincipalComponent"
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
	   val stateFilePath = this.getOptionalStringParam(appConfig, "state.filePath")
	   stateFilePath match {
	     case Some(path) => {
	       val compState = PrincipalCompState.load(path, fieldDelimOut).asScala.toMap
	     }
	     case None => {
	     }
	   }
	   
   }
   

}