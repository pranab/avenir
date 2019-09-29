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

object KmeansCluster extends JobConfiguration with GeneralUtility {
  
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
       val outputPrecision = getIntParamOrElse(appConfig, "output.precision", 3)	   
       val numIter = getIntParamOrElse(appConfig, "num.iter", 10)	
       val centroidShiftThreshold = getDoubleParamOrElse(appConfig, "centroid.shiftThreshold", .05)
       val clusterOutputPath = getMandatoryStringParam(appConfig, "cluster.outputPath", "missing cluster output path")
       
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
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
	   
	   //initilalize clusters
	   var allClusters = Map[(Int, Int), ArrayBuffer[ClusterData]]()
	   
	   //each cluster count
	   nClusters.foreach(nc => {
	     //each initial position
	     val clusters = ArrayBuffer[ClusterData]()
	     for (cg  <- 0 to numClustGroup - 1) {
	    	 val initClusters = data.takeSample(false, nc, 1).zipWithIndex
	    	 initClusters.foreach(cc => {
	    		 val cls = new ClusterData(nc, cg, cc._2, cc._1, fieldDelimIn) 
	    	     clusters += cls
	    	 })
	    	 allClusters += ((nc, cg) -> clusters)
	    	 activeCount += nc
	     }
	   })
	   
	   //iterate
	   for (i <- 1 to numIter if activeCount > 0) {
		   //within each cluster group, assign record to the nearest cluster
		   val clMemebers = data.flatMap(line => {
		     val fields = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		     val clMembers = ArrayBuffer[(ClusterData, Record)]()
		     allClusters.foreach(v => {
		       val (nc, cg) = v._1
		       val clusters = v._2
		       val allClusters = clusters.sortBy(c => c.getId()).toArray
		       
		       //real clusters
		       val realClusters = maxDist match {
		         case Some(mDist : Double) => {
		           //outlier tracking
		           allClusters.slice(0, allClusters.length-1)
		       	 }
		       	 case None => {
		       	   //normal case
		       	   allClusters
		       	 }
		       }
		     
		       //closest cluster
		       var clDist = realClusters.map(cl => {
		         var dist = cl.findDistaneToCentroid(fields, distanceFinder)
		         if (useDistRatio && i > 1) {
		           dist /= cl.getAvDistance()
		         }
		         (cl, line, dist)
		       }).sortBy(r => r._3).head
		       
		       clDist =  maxDist match {
		         case Some(mDist : Double) => {
		           //max dist and outlier tracking
		           if (i > 1 && clDist._3 > mDist)
		        	   (realClusters.last, clDist._2, clDist._3)
		           else
		             clDist
		       	 }
		       	 case None => {
		       	   //normal case
		       	   clDist
		       	 }
		       }
		       
		       val value = Record(2)
		       value.add(clDist._2, clDist._3)
		       clMembers += ((clDist._1, value))
		     })
		     
		     clMembers
		   })
		   
		   //adjust cluster centers
		   val createCluster = (value:Record) => {
		     val cl = new ClusterData()
		     cl.initMembership(attrOrdinals, schema)
		     val record = value.getString(0)
		     val distance = value.getDouble(1)
		     val fields = BasicUtils.getTrimmedFields(record, fieldDelimIn)
		     cl.addMember(fields, distance, schema, distSchema, distanceFinder)
		     cl
		   }
		   
		   //add to cluster
		   val addToCluster = (cl: ClusterData, value: Record) => {
		     val record = value.getString(0)
		     val distance = value.getDouble(1)
		     val fields = BasicUtils.getTrimmedFields(record, fieldDelimIn)
		     cl.addMember(fields, distance, schema, distSchema, distanceFinder)
		     cl
		   }
		   
		   //merge cluster
		   val mergeCluster = (clOne: ClusterData, clTwo: ClusterData) => {
			   clOne.merge(clTwo)
		   }
		   
		   //adjusted clusters
		   val adjustedClusters = clMemebers.combineByKey(createCluster, addToCluster, mergeCluster)
		   
		   //replace previous with current
		   val newClusters = adjustedClusters.map(r => {
		     val pClust = r._1
		     val cClust =  r._2
		     cClust.finishMemebership(pClust, distanceFinder, centroidShiftThreshold, outputPrecision,fieldDelimIn)
		     cClust
		   }).sortBy(c => (c.getNumClusterInGroup(), c.getGroupId())).cache
		   
		   //process results
		   allClusters = allClusters.empty
		   
		   val newClustersCol = newClusters.collect
		   newClustersCol.foreach(cl => {
		     val numClusterInGroup = cl.getNumClusterInGroup()
		     val groupId = cl.getGroupId()
		     val clusters = allClusters.getOrElse((numClusterInGroup, groupId), ArrayBuffer[ClusterData]())
		     clusters += cl
		     if (cl.isActive())
		       activeCount += 1
		   })
		   
		   //termination
		   if (i == numIter || activeCount == 0) {
			   //find best i.e group with the lowest average sse
			   val bestClusters = newClusters.map(c => {
			     ((c.getNumClusterInGroup(), c.getGroupId()), (c.getSse(), 1))
			   }).reduceByKey((v1, v2) => {
			     (v1._1 + v2._1, v1._2 + v2._2)
			   }).mapValues(v => {
			     v._1 / v._2
			   }).sortBy(r => r._2, true, 1).cache
			   
			   //detailed cluster output
			   val serClusters = newClusters.map(c => {
				   c.withFieldDelim(fieldDelimOut).withOutputPrecision(outputPrecision).toString
			   }).cache
			   
			   if (debugOn) {
			       bestClusters.foreach(r => println(r))
			       serClusters.foreach(r => println(r))
			   }
			     
			   if (saveOutput) {
			     bestClusters.saveAsTextFile(outputPath) 
				 serClusters.saveAsTextFile(clusterOutputPath) 
			   }
		   } else {
		     newClusters.unpersist(false)
		   }
	   }
	   
   }
}