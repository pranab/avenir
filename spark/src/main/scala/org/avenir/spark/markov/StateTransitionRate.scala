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

package org.avenir.spark.markov

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext

object StateTransitionRate extends JobConfiguration {
  
 /**
 * @param args
 * @return
 */
   def main(args: Array[String]) {
	   val Array(master: String, inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf("stateTransitionRate", config)
	   val sparkCntxt = new SparkContext(sparkConf)
	   
	   val fieldDelimIn = config.getString("app.field.delim.in")
	   val fieldDelimOut = config.getString("app.field.delim.out")
	   
	   
	   
   }  
}