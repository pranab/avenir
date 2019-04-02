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


package org.avenir.spark.sequence

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.util.BasicUtils
import org.chombo.spark.common.GeneralUtility
import org.chombo.spark.common.Record
import scala.collection.mutable.ArrayBuffer
import org.chombo.util.TabularData

/**
* Pair wise sub sequence matching
* @param args
* @return
*/
object DotMatrixSubSequenceMatching extends JobConfiguration with GeneralUtility {

   /**
 * @param args
 * @return
 */
   def main(args: Array[String]) {
	   val appName = "normalizer"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)

	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delimIn", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delimOut", ",")
	   val seqDelim = getStringParamOrElse(appConfig, "seq.delim", "\\s+")
	   val windowLength = getIntParamOrElse(appConfig, "window.length", 2)
	   val numBuckets = getIntParamOrElse(appConfig, "num.buckets", 16)
	   val buckets  = List.range(0, numBuckets)
	   
	   val scoreMatrixFilePath = getOptionalStringParam(appConfig, "score.matrixFilePath")
	   val scoreConfig = scoreMatrixFilePath match {
	     case Some(path) => {
	       val fiStrm = BasicUtils.getFileStream(path)
	       val scoreTable = new TabularData(fiStrm)
	       val scoreThreshold = this.getMandatoryIntParam(appConfig, "score.threshold", "")
	       Some(scoreTable, scoreThreshold)
	     }
	     case None => None
	   }
	   val outputAlignments = this.getBooleanParamOrElse(appConfig, "output.alignments", false)
	   val debugOn = appConfig.getBoolean("debug.on")
	   val saveOutput = appConfig.getBoolean("save.output")
	   
	   //input
	   val data = sparkCntxt.textFile(inputPath)
	   
	   //key with all bucket pairs and record as value
	   val bucketedData = data.flatMap(line => {
		   val items = BasicUtils.getTrimmedFields(BasicUtils.splitOnFirstOccurence(line, fieldDelimIn, true))
		   val keyRec = Record(1)
		   keyRec.addString(items(0))
		   val hash = keyRec.hashCode
		   val thisBucket = (if (hash < 0) -hash else hash) % numBuckets
		   var bucketId = 0
		   val bucketedRec = buckets.map(b => {
		     val bucketPairHash = if (thisBucket > b) {
		        bucketId = 0
		        thisBucket << 12 | b
		     } else { 
		        bucketId = 1
		        b << 12 | thisBucket  
		     }
		       
		     (bucketPairHash, (bucketId,line)) 
		   })
		   
		   bucketedRec
	   })
	   
	   
	   //group by key and generate distances
	   val pairScores  = bucketedData.groupByKey().flatMapValues(recs => {
	     val firstBucket = recs.filter(r => r._1 == 0)
	     val secondBucket = recs.filter(r => r._1 == 1)
	     val pairScores = ArrayBuffer[Record]()
	     
	     //first bucket
	     firstBucket.foreach(f => {
	       val firstRec = f._2
	       val firstItems = BasicUtils.getTrimmedFields(BasicUtils.splitOnFirstOccurence(firstRec, fieldDelimIn, true))
	       val firstKey = firstItems(0)
	       val firstSeq = firstItems(1)
	       
	       //second bucket
	       secondBucket.foreach(s => {
	    	   val secondRec = s._2
	    	   val secondItems = BasicUtils.getTrimmedFields(BasicUtils.splitOnFirstOccurence(secondRec, fieldDelimIn, true))
	           val secondKey = secondItems(0)
	           val secondSeq = secondItems(1)
	           //val score = findScore(firstSeq, secondSeq, seqDelim, windowLength)
	           
	           val dotMatrix = scoreConfig match {
	    	     case Some((scoreTable, scoreThreshold)) => 
	    	       findScore(firstSeq, secondSeq, seqDelim, windowLength, scoreTable, scoreThreshold)
	    	     case None => 
	    	       findScore(firstSeq, secondSeq, seqDelim, windowLength)
	    	   }
	           val score = dotMatrix.getSum()
	           if (outputAlignments) {
		           val scoreRec = Record(6)
		           scoreRec.addString(firstKey)
		           scoreRec.addString(secondKey)
		           scoreRec.addInt(score)
		           val firstSeqLen = BasicUtils.getTrimmedFields(firstSeq, seqDelim).length
		           val secondSeqLen = BasicUtils.getTrimmedFields(secondSeq, seqDelim).length
		           val alignments = dotMatrix.withDeilmeter(seqDelim).toString()
		           scoreRec.addInt(firstSeqLen)
		           scoreRec.addInt(secondSeqLen)
		           scoreRec.addString(alignments)
		           pairScores += scoreRec
	           } else {
		           val scoreRec = Record(3)
		           scoreRec.addString(firstKey)
		           scoreRec.addString(secondKey)
		           scoreRec.addInt(score)
		           pairScores += scoreRec
	           }
	           
	       })
	       
	     })
	     pairScores
	   }).values
	   
	   
       if (debugOn) {
         val records = pairScores.collect.sliding(10)
         records.foreach(r => println(r))
       }
	   
	   if(saveOutput) {	   
	     pairScores.saveAsTextFile(outputPath) 
	   }

   }
   
   /**
   * @param firstSeq
   * @param secondSeq
   * @param seqDelim
   * @param windowLength
   * @return
   */
   def findScore(firstSeq:String, secondSeq:String, seqDelim:String, windowLength:Int) : TabularData = {
     val firstItems = BasicUtils.getTrimmedFields(firstSeq, seqDelim)
     val secondItems = BasicUtils.getTrimmedFields(secondSeq, seqDelim)
     val window = new Array[String](2)
     val dotMatrix = new TabularData(firstItems, secondItems)
     dotMatrix.setAll(-1)
     
     //first seq as rows
     for (i <- 0 to firstItems.length - windowLength) {
       //match in second seq
       Array.copy(firstItems, i, window, 0, windowLength)
       val positions = BasicUtils.findStringSubSequencePositions(secondItems, window)
       
       //each match
       positions.foreach(pos => {
         var row = i
         var col = pos
         
         //match all window elements
         for (j <- 0 to windowLength) {
        	 dotMatrix.set(row, col, 1)
        	 row += 1
        	 col += 1
         }
       })
       
     }
     dotMatrix
   }
   
   /**
   * @param firstSeq
   * @param secondSeq
   * @param seqDelim
   * @param windowLength
   * @return
   */
   def findScore(firstSeq:String, secondSeq:String, seqDelim:String, windowLength:Int, scoreTable: TabularData, 
       scoreThreshold:Int) : TabularData = {
     val firstItems = BasicUtils.getTrimmedFields(firstSeq, seqDelim)
     val secondItems = BasicUtils.getTrimmedFields(secondSeq, seqDelim)
     val window = new Array[String](2)
     val dotMatrix = new TabularData(firstItems, secondItems)
     dotMatrix.setAll(-1)
     
     //first seq as rows
     var score = 0
     for (i <- 0 to firstItems.length - windowLength) {
       //match in second seq
       Array.copy(firstItems, i, window, 0, windowLength)
       val positions = findStringSubSequencePositions(secondItems, window, scoreTable, scoreThreshold)
       
       //each match
       positions.foreach(pos => {
         score += pos._2
         
         //match all window elements
         var row = i
         var col = pos._1
         for (j <- 0 to windowLength) {
        	 dotMatrix.set(row, col, 1)
        	 row += 1
        	 col += 1
         }
         
       })
       
     }
     dotMatrix
   }

  /**
  * @param source
  * @param window
  * @param scoreTable
  * @param scoreThreshold
  * @return
  */
  def findStringSubSequencePositions(source: Array[String], window: Array[String],scoreTable: TabularData, 
       scoreThreshold:Int) : Array[(Int, Int)] =  {
     val lastPos = source.length - window.length
     val positions = ArrayBuffer[(Int, Int)]()
     
     var score = 0
     var i = 0
     while (i < lastPos) {
       for (j <- 0 to window.length) {
         val s = source(i + j)
         val w = window(j)
         score += scoreTable.get(s, w)
       }
       
       if (score >= scoreThreshold) {
         val posScore = (i, score)
         positions += posScore
         i += window.length
       } else {
         i += 1
       }
     }
     
     positions.toArray
   }
}