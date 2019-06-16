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
import org.chombo.spark.common.Record
import org.chombo.util.BasicUtils
import org.avenir.util.StateTransitionProbability
import org.apache.spark.rdd.RDD
import org.chombo.spark.common.GeneralUtility
import scala.collection.mutable.ArrayBuffer

/**
 * generates Markov state transition probability matrix for data with or without class labels
 * Handles multiple entities. Creates Markov state transition probability matrix for each entity.
 * @param args
 * @return
 */
object MarkovStateTransitionModel extends JobConfiguration with GeneralUtility {
  
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "markovStateTransitionModel"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyFieldOrdinals = toIntArray(getMandatoryIntListParam(appConfig, "id.field.ordinals"))
	   val classAttrOrdinal = getOptionalIntParam(appConfig, "class.attr.ordinal")
	   val seqStartOrdinal = getMandatoryIntParam(appConfig, "seq.start.ordinal")
	   val states = getMandatoryStringListParam(appConfig, "state.list", "missing state list")
	   val statesArr = BasicUtils.fromListToStringArray(states)
	   val scale = getMandatoryIntParam(appConfig, "trans.prob.scale")
	   val outputPrecision = getIntParamOrElse(appConfig, "output.precision", 3);
	   val seqLongFormat = getBooleanParamOrElse(appConfig, "data.seqLongFormat", false)
	   val seqFieldOrd = if (seqLongFormat) getMandatoryIntParam(appConfig, "seq.field.ordinal") else -1
	   val stateFieldOrd = if (seqLongFormat) getMandatoryIntParam(appConfig, "state.field.ordinal") else -1
	   val mergeKeys = getBooleanParamOrElse(appConfig, "data.mergeKeys", false)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   //read input
	   val data = sparkCntxt.textFile(inputPath)
	   
	   val keyedStatePairs = if (seqLongFormat) {
	     keyedStatePairForLongFormat(data, fieldDelimIn, classAttrOrdinal, keyFieldOrdinals, seqFieldOrd,  stateFieldOrd, mergeKeys)
	   } else {
	     keyedStatePairForCompactFormat(data, fieldDelimIn, seqStartOrdinal,classAttrOrdinal, keyFieldOrdinals)
	   }
	   
	   //key value records
	   val x = data.flatMap(line => {
		   val items = line.split(fieldDelimIn, -1)
		   val seqValIndexes = List.range(seqStartOrdinal+1, items.length)
		   
		   val stateTrans = seqValIndexes.map(idx => {
			   val keyRec = classAttrOrdinal match {
			   		case Some(classOrd:Int) => {
			   			//with class attribute
			   			val classVal = items(classOrd)
			   			val keyRec = Record(keyFieldOrdinals.length + 3, items, keyFieldOrdinals)
			   			keyRec.addString(classVal)
			   			keyRec
			   		}
		     
			   		case None => {
			   			//without class attribute
			   			val keyRec = Record(keyFieldOrdinals.length + 2, items, keyFieldOrdinals)
			   			keyRec
			   		}
			   }
			   keyRec.addString(items(idx-1)).addString(items(idx))
			   (keyRec, 1)
		     
		   	})
		   stateTrans
	   }).reduceByKey(_ + _)
	   
	   //move state pairs from key to value
	   val transData = keyedStatePairs.map(kv => {
	     //key: id and optional class value
	     val size = kv._1.size
	     val newKeyRec = Record(kv._1, 0, size-2)
	     
		 //value: state pairs from key  and count from value
	     val newValRec = Record(3)
	     newValRec.add(kv._1, size-2, size)
		 newValRec.add(kv._2)
		 (newKeyRec, newValRec)
	   })
	   
	   //group by key and map values to convert sate transition matrix
	   val transProb = transData.groupByKey().mapValues(stc => {
	     val stTransProb = new StateTransitionProbability(statesArr, statesArr)
	     stTransProb.withScale(scale).withFloatPrecision(outputPrecision)
	     stc.foreach(c => {
	       stTransProb.add(c.getString(0), c.getString(1), c.getInt(2))
	     })
	     stTransProb.normalizeRows()
	     stTransProb
	   })
	   
	   if (debugOn) {
	     val colTransProb = transProb.collect
	     colTransProb.foreach(s => {
	       println("id:" + s._1)
	       println("state trans probability:" + s._2)
	     })
	   }

	   if (saveOutput) {
	     transProb.saveAsTextFile(outputPath)
	   }

   }
   
   /**
   * @param config
   * @param paramName
   * @param defValue
   * @param errorMsg
   * @return
   */
   def keyedStatePairForCompactFormat(data:RDD[String], fieldDelimIn:String, seqStartOrdinal:Int,
       classAttrOrdinal:Option[Int], keyFieldOrdinals:Array[Int]) : RDD[(Record,Int)] = {
	   data.flatMap(line => {
		   val items = line.split(fieldDelimIn, -1)
		   val seqValIndexes = List.range(seqStartOrdinal+1, items.length)
		   
		   val stateTrans = seqValIndexes.map(idx => {
			   val keyRec = classAttrOrdinal match {
			   		case Some(classOrd:Int) => {
			   			//with class attribute
			   			val classVal = items(classOrd)
			   			val keyRec = Record(keyFieldOrdinals.length + 3, items, keyFieldOrdinals)
			   			keyRec.addString(classVal)
			   			keyRec
			   		}
		     
			   		case None => {
			   			//without class attribute
			   			val keyRec = Record(keyFieldOrdinals.length + 2, items, keyFieldOrdinals)
			   			keyRec
			   		}
			   }
			   keyRec.addString(items(idx-1)).addString(items(idx))
			   (keyRec, 1)
		     
		   	})
		   stateTrans
	   }).reduceByKey(_ + _)
   }
   
   /**
  * @param data
  * @param fieldDelimIn
  * @param classAttrOrdinal
  * @param keyFieldOrdinals
  * @param seqFieldOrd
  * @param stateFieldOrd
  * @return
  */
  def keyedStatePairForLongFormat(data:RDD[String], fieldDelimIn:String, classAttrOrdinal:Option[Int], 
       keyFieldOrdinals:Array[Int], seqFieldOrd:Int,  stateFieldOrd:Int, mergeKeys:Boolean) : RDD[(Record,Int)] = {
    data.map(line => {
    	 val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
    	 val keyRec = classAttrOrdinal match {
	   		case Some(classOrd:Int) => {
	   			//with class attribute
	   			val classVal = items(classOrd)
	   			val keyRec = Record(keyFieldOrdinals.length + 1, items, keyFieldOrdinals)
	   			keyRec.addString(classVal)
	   			keyRec
	   		}
     
	   		case None => {
	   			//without class attribute
	   			val keyRec = Record(keyFieldOrdinals.length, items, keyFieldOrdinals)
	   			keyRec
	   		}
		  }
    	 val valRec = Record(2)
    	 valRec.addLong(items(seqFieldOrd).toLong)
    	 valRec.addString(items(stateFieldOrd))
    	 (keyRec, valRec)
     }).groupByKey.flatMap(r => {
       val key = r._1
       val values = r._2.toArray
       val sortedValues = values.sortWith((v1, v2) => v1.getLong(0) < v2.getLong(0))
       val statePairs = ArrayBuffer[Record]()
       for (i <- 1 to sortedValues.length - 1) {
         val pair = Record(2)
         pair.add(sortedValues(i-1).getString(1), sortedValues(i).getString(1))
         statePairs += pair
       }
       statePairs.map(v => {
         val newKey = if (mergeKeys) Record(Record("all"), v) else Record(key, v)
         (newKey, 1)
       })
     }).reduceByKey((v1, v2) => v1 + v2)
   }
}