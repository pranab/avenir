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
import org.avenir.cluster.Cluster
import org.chombo.spark.common.GeneralUtility
import org.chombo.spark.common.SeasonalUtility
import org.apache.commons.math3.ml.distance.EuclideanDistance
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer
import org.apache.commons.math3.ml.clustering.DoublePoint
import org.avenir.cluster.ClusterUtility

/**
* KMeans plus plus clustering
* @author pranab
*
*/
object KMeansPlusPlusCluster extends JobConfiguration with GeneralUtility with SeasonalUtility {
  
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "kMeansPlusPlusCluster"
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
	   val numClusters = toIntArray(getMandatoryIntListParam(appConfig, "num.clusters",  "missing cluster count list"))
	   val outputPrecision = getIntParamOrElse(appConfig, "output.precision", 3)	   
       val numIter = getIntParamOrElse(appConfig, "num.iter", 10)	
       val clusterOutputPath = getMandatoryStringParam(appConfig, "cluster.outputPath", "missing cluster output path")
       val seasonalAnalysis = getBooleanParamOrElse(appConfig, "seasonal.analysis", false)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   val seasonalAnalyzers = creatOptionalSeasonalAnalyzerArray(this, appConfig, seasonalAnalysis)
	   val keyLen = keyFieldOrdinals.length + 2 
	   val dimension = attrOrdinals.length
	   
	   //input
	   val data = sparkCntxt.textFile(inputPath).cache
	   val clustData = data.flatMap(line => {
		   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		   val keyRec = Record(keyLen + 1, items, keyFieldOrdinals)
		   addSeasonalKeys(seasonalAnalyzers, items, keyRec)
		   numClusters.map(v => {
		     keyRec.addInt(v)
		     (keyRec, items)
		   })
	   }).groupByKey.map(r => {
	     val keyRec = r._1
	     val nClust = keyRec.getInt(keyRec.size - 1)
	     val values = r._2.toArray
	     val records   = BasicUtils.extractFieldsAsDoubleArrayList(values, attrOrdinals)
	     val distMeas = new EuclideanDistance()
	     
	       
	       //different initializations
	       val clusters = (0 to numClustGroup - 1).map(i => {
	         val clusterer = new KMeansPlusPlusClusterer[DoublePoint](nClust, numIter,distMeas)
	         val clustCentroids = clusterer.cluster(records)
	         val clusters = ClusterUtility.getClusterProperies(clustCentroids,  distMeas)
	         val avSse = ClusterUtility.getAverageSse(clusters)
	         (clusters, avSse)
	       }).toArray.sortWith((v1, v2) => v1._2 < v2._2)
	       
	       (keyRec,clusters(0))
	   })
	   
	   //move cluster count from key to value and find optimum number of clusters
	   val serClustData = clustData.map(r => {
	     val keyRec = Record(r._1, 0, r._1.size - 1)
	     (keyRec, r._2)
	   }).groupByKey.map(r => {
	     val keyRec = r._1
	     
	     //find knuckle point i.e max second difference
	     val sortedClusterLists = r._2.toArray.sortWith((v1, v2) => v1._1.length < v2._1.length)
	     var knucklePt = 0
	     var maxSecDiff = 0.0
	     for (i <- 1 to sortedClusterLists.size - 1) {
	       val secDiff = Math.abs(sortedClusterLists(i+1)._2 - 2 * sortedClusterLists(i)._2 + sortedClusterLists(i-1)._2)
	       if (secDiff > maxSecDiff) {
	         maxSecDiff = secDiff
	         knucklePt = i
	       }
	     }
	     
	     val selClusterList = sortedClusterLists(knucklePt)
	     val stBld = new StringBuilder(keyRec.toString(fieldDelimOut))
	     	stBld.append(fieldDelimOut).append(dimension)
	     selClusterList._1.foreach(c => {
	       stBld.append(fieldDelimOut).append(c.withFieldDelim(fieldDelimOut).withOutputPrecision(outputPrecision).toString())
	     })
	     stBld.toString
	   })
	   
	   if (debugOn) {
	     val colSerClustData = serClustData.collect.slice(0, 50)
	     colSerClustData.foreach(s => println("state trans probability:" + s))
	   }

	   if (saveOutput) {
	     serClustData.saveAsTextFile(outputPath)
	   }

   }

}