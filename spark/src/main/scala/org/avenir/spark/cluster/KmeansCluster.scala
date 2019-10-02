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
import org.apache.spark.rdd.RDD
import org.chombo.util.BasicUtils
import org.chombo.spark.common.Record
import scala.collection.mutable.ArrayBuffer
import org.chombo.distance.InterRecordDistance
import org.avenir.cluster.ClusterData
import org.chombo.spark.common.GeneralUtility
import org.chombo.spark.common.SeasonalUtility

/**
* KMeans  clustering supporting categorical data type
* @author pranab
*
*/
object KmeansCluster extends JobConfiguration with GeneralUtility with SeasonalUtility {
  
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "kmeansCluster"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //config params
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyFieldOrdinals = toIntArray(getMandatoryIntListParam(appConfig, "id.fieldOrdinals", ""))
	   val attrOrdinals = toIntArray(getMandatoryIntListParam(config, "attr.ordinals", "missing attribute field"))
	   val seqFieldOrd = getMandatoryIntParam(appConfig, "seq.field.ordinal","missing sequence field ordinal") 
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
       val outputPrecision = getIntParamOrElse(appConfig, "output.precision", 3)	   
       val numIter = getIntParamOrElse(appConfig, "num.iter", 10)	
       val centroidShiftThreshold = getDoubleParamOrElse(appConfig, "centroid.shiftThreshold", .05)
       val clusterOutputPath = getMandatoryStringParam(appConfig, "cluster.outputPath", "missing cluster output path")
       val seasonalAnalysis = getBooleanParamOrElse(appConfig, "seasonal.analysis", false)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   val seasonalAnalyzers = creatOptionalSeasonalAnalyzerArray(this, appConfig, seasonalAnalysis)
	   val keyLen = keyFieldOrdinals.length + (if (seasonalAnalysis) 2 else 0) + 2
	   var activeCount = 0
	   var outlierTracking = false
       
	   val nClusters = maxDist match {
	     case Some(mDist : Double) => {
	       //extra cluster for collecting outliers
	       outlierTracking =  true
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
	   val data = sparkCntxt.textFile(inputPath).cache
	   
	   //initilalize clusters keyed by number of clusters and initial positions
	   val allKeys = ArrayBuffer[Record]()
	   
	   //each cluster count
	   nClusters.foreach(nc => {
	     val clusters = ArrayBuffer[ClusterData]()
	     
	     //each initial cluster  position
	     for (cg  <- 0 to numClustGroup - 1) {
	    	 val initClusters = data.takeSample(false, nc, 1)
	    	 val size = 2 + initClusters.length
	    	 val keyRec = Record(size, initClusters, 2)
	    	 keyRec.addInt(0, nc)
	    	 keyRec.addInt(1, cg)
	    	 allKeys += keyRec
	     }
	   })
	   
	   
	   val x = data.flatMap(line => {
		 val fields = BasicUtils.getTrimmedFields(line, fieldDelimIn)
	     allKeys.map(k => {
	       val size = keyFieldOrdinals.length +  k.size
	       val keyRec = Record(size, k, keyFieldOrdinals.length)
	       keyRec.initialize
	       Record.populateFields(fields, keyFieldOrdinals, keyRec)
	       (keyRec, line)
	     })
	   }).groupByKey.map(r => {
	     val keyRec = r._1
	     val values = r._2.toArray
	     var offset = keyFieldOrdinals.length
	     val nc = keyRec.getInt(offset)
	     offset += 1
	     val cg = keyRec.getInt(offset)
	     offset += 1
	     val clCenters = Record.extractFields(keyRec, offset, keyRec.size)
	     val clusters = clCenters.zipWithIndex.map(cc => {
	       val cl = new ClusterData(nc, cg, cc._2, cc._1, fieldDelimIn,  schema,  distSchema,  distanceFinder,
			 centroidShiftThreshold) 
	       cl.initMembership(attrOrdinals)
	       cl
	     })
	     
	     values.foreach(line => {
	       val fields = BasicUtils.getTrimmedFields(line, fieldDelimIn)
	       
	       //closest cluster for a record
		   val clDist = clusters.map(cl => {
		     var dist = cl.findDistaneToCentroid(fields, distanceFinder)
		     if (useDistRatio) {
		           dist /= cl.getAvDistance()
		     }
		     (cl, line, dist)
		   }).sortBy(r => r._3).head
		   
		   //add member
		   clDist._1.addMember(fields, clDist._3)
		   
	     })
	     
	     
	     
	   })
	   
	   
	   
   }
}