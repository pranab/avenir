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

package org.avenir.spark.interpret

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.util.BasicUtils
import org.chombo.spark.common.Record
import org.chombo.util.BaseAttribute
import com.typesafe.config.Config
import org.chombo.spark.common.GeneralUtility
import scala.collection.mutable.ArrayBuffer
import org.avenir.util.Prediction

/**
* Black box Machine learning interpretation by Individual Conditional Expectation (ICE)
* @return
*/
object IndividualConditionalExpectation extends JobConfiguration with GeneralUtility {
   case class Feature(dataType: String, stepSize: Double, numSteps: Int, direction:Int, cardinality: Int, encoded:Boolean)

   /**
   * @param args
   * @return
   */
   def main(args: Array[String]) {
	   val appName = "outlierCounter"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configuration params
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyLen = getMandatoryIntParam(appConfig, "data.keyLen", "missing key length")
	   val precision = getIntParamOrElse(appConfig, "output.precision", 3)
	   
	   //feature traversal data
	   val featureOrdinals = toIntList(getMandatoryIntListParam(appConfig, "feature.ordinals", 
	       "missing exploratory feature ordinls"))
	   var featureGrid = Map[Int, Feature]()
	   featureOrdinals.foreach(i => {
	     val key = "feature." + i
	     val feConfig = appConfig.getConfig(key)
	     val feType = getMandatoryStringParam(feConfig, "type", "missing feature type")
	     feType match {
	       case BaseAttribute.DATA_TYPE_INT => {
	         val step = getMandatoryIntParam(feConfig, "step", "missing step size")
	         val numSteps = getMandatoryIntParam(feConfig, "numSteps", "missing number of steps")
	         val direction = getMandatoryIntParam(feConfig, "direction", "missing direction")
	         featureGrid += (i -> Feature(feType, step.toDouble, numSteps, direction, 0, false))
	       }
	       case BaseAttribute.DATA_TYPE_DOUBLE => {
	         val step = getMandatoryDoubleParam(feConfig, "step", "missing step size")
	         val numSteps = getMandatoryIntParam(feConfig, "numSteps", "missing number of steps")
	         val direction = getMandatoryIntParam(feConfig, "direction", "missing direction")
	         featureGrid += (i -> Feature(feType, step, numSteps, direction, 0, false))
	       }
	       case BaseAttribute.DATA_TYPE_CATEGORICAL => {
	         val cardinality = getMandatoryIntParam(feConfig, "cardinality", "missing cardinality")
	         val encoded = getMandatoryBooleanParam(feConfig, "encoded", "missing encoded")
	         BasicUtils.assertCondition(encoded, "only binary encoded categorical supported")
	         featureGrid += (i -> Feature(feType, 0, 0, 0, cardinality, encoded))
	       }
	     }
	   })
	   
	   val refRecTag = "R"
	   val genRecTag = "G"
	   val predictionUrl = getMandatoryStringParam(appConfig, "prediction.url", "missing prediction service URL")
	   val predictionFieldDelim = getStringParamOrElse(appConfig, "prediction.reqFieldDelim", ",")
	   val predictionRecDelim = getStringParamOrElse(appConfig, "prediction.reqRecDelim", ",,")
	   val sortByPrediction = getBooleanParamOrElse(appConfig, "prediction.sortDescending", true)
	   
	   val debugOn = appConfig.getBoolean("debug.on")
	   val saveOutput = appConfig.getBoolean("save.output")
	   
	   //input
	   val data = sparkCntxt.textFile(inputPath)
	   
	   //generate neighborhood records
	   val genRecs = data.flatMap(line => {
   		   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
   		   val refRec = line +  fieldDelimIn + refRecTag
   		   val recs = Array[String](line)
   		   genCandidates(recs, fieldDelimIn, featureOrdinals.toArray, 0, featureGrid, genRecTag:String, precision, refRec)
	   })	
	   
	   //group by key and get predictions
	   val recsWithPrediction = genRecs.map(line => {
   		   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
   	       val keyRec = Record(items, 0, keyLen)
   	       (keyRec, line)
	   }).groupByKey.flatMap(r => {
	     val recs = r._2.toArray
	     val reqMsgArr = ArrayBuffer[String]()
	     for (rec <- recs) {
	       val reqRec = BasicUtils.getTrimmedFields(rec, fieldDelimIn).slice(keyLen, rec.length - 2).
	    		   mkString(predictionFieldDelim)
	       reqMsgArr += reqRec
	     }
	     val reqMsg = reqMsgArr.mkString(predictionRecDelim)
	     val reqJson = "{\"recs\":\"" +  reqMsg + "\"}"
	     val respJson = BasicUtils.httpJsonPost(predictionUrl, reqJson)
	     val prediction = Prediction.decodeJson(respJson)	   
	     var recsWithPred = recs.zip(prediction.getPredictions()).toList.map(r => {r._1 + fieldDelimOut + r._2})
	     if(sortByPrediction) {
	       recsWithPred = recsWithPred.sortWith((r1, r2) => {
	         val f1 = BasicUtils.getTrimmedFields(r1, fieldDelimIn)
	         val f2 = BasicUtils.getTrimmedFields(r2, fieldDelimIn)
	         f1(f1.length-1).toDouble > f2(f2.length-1).toDouble
	       })
	     }
	     recsWithPred
	   })
	   
       if (debugOn) {
         val records = recsWithPrediction.collect.slice(0, 19)
         records.foreach(r => println(r))
       }
	   
	   if(saveOutput) {	   
	     recsWithPrediction.saveAsTextFile(outputPath) 
	   }
	   
   }
     
   
  /**
  * @param recs
  * @param fieldDelimIn
  * @param features
  * @param curFeature
  * @param featureGrid
  * @param genRecTag
  * @param refRec
  * @return
  */
  def genCandidates(recs:Array[String], fieldDelimIn:String, features:Array[Int], curFeature:Int, 
       featureGrid:Map[Int, Feature], genRecTag:String, precision:java.lang.Integer, refRec:String) : Array[String] =  {
	 val newRecs = ArrayBuffer[String]()
	 
     for (rec <- recs) {
       val items = BasicUtils.getTrimmedFields(rec, fieldDelimIn)
       val featTrav= featureGrid.get(curFeature).get
       featTrav.dataType match {
         //int
         case BaseAttribute.DATA_TYPE_INT => {
           val fIndx = features(curFeature)
           val ref = items(fIndx).toInt
           val grid = BasicUtils.traverse(ref, featTrav.stepSize.toInt, featTrav.numSteps, featTrav.direction)
           for (grVal <- grid) {
             var newRec = new String(rec)
             val items = BasicUtils.getTrimmedFields(newRec, fieldDelimIn)
             items(fIndx) = grVal.toString
             newRec =  items.mkString(fieldDelimIn) + fieldDelimIn + genRecTag
             newRecs += newRec
           }
         }
         //double
         case BaseAttribute.DATA_TYPE_DOUBLE => {
           val fIndx = features(curFeature)
           val ref:java.lang.Double = items(fIndx).toDouble
           val grid = BasicUtils.traverse(ref, featTrav.stepSize, featTrav.numSteps, featTrav.direction)
           for (grVal <- grid) {
             var newRec = new String(rec)
             val items = BasicUtils.getTrimmedFields(newRec, fieldDelimIn)
             items(fIndx) = BasicUtils.formatDouble(grVal, precision)
             newRec =  items.mkString(fieldDelimIn) + fieldDelimIn + genRecTag
             newRecs += newRec
           }
           
         }
         //categorical
         case BaseAttribute.DATA_TYPE_CATEGORICAL => {
           val fIndx = features(curFeature)
           for (i <- 0 to featTrav.cardinality-1){
             var newRec = new String(rec)
             val items = BasicUtils.getTrimmedFields(newRec, fieldDelimIn)
             val arr = BasicUtils.initIntArray(featTrav.cardinality, 0)
             arr(i) = 1
             for (j <- 0 to featTrav.cardinality-1) {
               items(fIndx + j) = arr(j).toString
             }
             newRec =  items.mkString(fieldDelimIn) + fieldDelimIn + genRecTag
             newRecs += newRec
           }
         
         }
       }
     }
	 
     if (curFeature < features.length - 1) {
       //not done recurse
       genCandidates(newRecs.toArray, fieldDelimIn, features, curFeature+1, featureGrid, genRecTag, precision, refRec)
     } else {
       //done
       newRecs += refRec
       return newRecs.toArray
     }
   }
   
   
}