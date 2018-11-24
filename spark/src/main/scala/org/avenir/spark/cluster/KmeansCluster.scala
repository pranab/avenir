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

package org.avenir.spark.cluster

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import com.typesafe.config.Config
import org.chombo.util.BasicUtils
import org.chombo.spark.common.Record
import scala.collection.mutable.ArrayBuffer
import org.chombo.distance.InterRecordDistance
import org.avenir.cluster.Cluster

object KmeansCluster extends JobConfiguration {
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
	   val fieldDelimIn = appConfig.getString("field.delim.in")
	   val fieldDelimOut = appConfig.getString("field.delim.out")
	   val attrOrdinals = getMandatoryIntListParam(config, "attr.ordinals", "missing attribute field").
	   	asScala.toArray.map(v => v.toInt)
	   val numClustGroup = getIntParamOrElse(appConfig, "num.clustGroup", 10)
	   val numClusters = getMandatoryIntListParam(appConfig, "num.clusters",  "missing cluster count list").asScala.toList
	   val useDistRatio = getBooleanParamOrElse(appConfig, "use.distRatio", true)
	   val maxDist = getOptionalDoubleParam(appConfig, "max.dist")
	   val schemaPath = getMandatoryStringParam(appConfig, "schema.path", "missing schema file path")
	   val schema = BasicUtils.getGenericAttributeSchema(schemaPath)
	   val distSchemaPath = getMandatoryStringParam(appConfig, "dist.schemaPath", "missing distance schema file path")
	   val distSchema = BasicUtils.getDistanceSchema(distSchemaPath)
       val distanceFinder = new InterRecordDistance(schema, distSchema, fieldDelimIn);
       distanceFinder.withFacetedFields(attrOrdinals);
	   
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   val nClusters = maxDist match {
	     case Some(mDist : Double) => {
	       //extra cluster for collecting outliers
	       val nClusters = numClusters.map(v => v + 1)
	       nClusters
	     }
	     case None => {
	       //normal 
	       val nClusters = numClusters.map(v => v.toInt)
	       nClusters
	     }
	   }
	   
	   //input
	   val data = sparkCntxt.textFile(inputPath)
	   
	   //initilalize clusters
	   var allClusters = Map[Int, List[Cluster]]()
	   
	   //each cluster count
	   nClusters.foreach(nc => {
	     //each initial position
	     val clusters = ArrayBuffer[Cluster]()
	     for (cg  <- 0 to numClustGroup - 1) {
	    	 val initClusters = data.takeSample(false, nc, 1).zipWithIndex
	    	 initClusters.foreach(cc => {
	    		 val cls = new Cluster(nc, cg, cc._2, cc._1) 
	    	     clusters += cls
	    	 })
	     }
	     allClusters += (nc -> clusters.toList)
	   })
	   
	   
	   //within each cluster group, assign data to the nearest cluster
	   data.flatMap(line => {
	     
	     allClusters.foreach(v => {
	       val nc = v._1
	       val clusters = v._2
	       clusters.map(cl => {
	       })
	     })
	     
	     List()
	   })
	   
	   
   }
}