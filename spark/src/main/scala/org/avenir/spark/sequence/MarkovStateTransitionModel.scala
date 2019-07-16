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
	   val seqLongFormat = getBooleanParamOrElse(appConfig, "data.seqLongFormat", false)
	   val seqStartOrdinal = getConditionalMandatoryIntParam(!seqLongFormat, appConfig, "seq.start.ordinal", 
	       "missing sequence start ordinal for compact format")
	   val states = getMandatoryStringListParam(appConfig, "state.list", "missing state list")
	   val statesArr = BasicUtils.fromListToStringArray(states)
	   val scale = getIntParamOrElse(appConfig, "trans.prob.scale", 1)
	   val outputPrecision = getIntParamOrElse(appConfig, "output.precision", 3);
	   val seqFieldOrd = getConditionalMandatoryIntParam(seqLongFormat, appConfig, "seq.field.ordinal", 
	       "missing sequence field ordinal") 
	   val stateFieldOrd = getConditionalMandatoryIntParam(seqLongFormat, appConfig, "state.field.ordinal", 
	       "missing state field ordinal") 
	   val mergeKeys = getBooleanParamOrElse(appConfig, "data.mergeKeysNeeded", false)
	   val laplaceCorr = getBooleanParamOrElse(appConfig, "data.laplaceCorrNeeded", false)
	   val outputCompact = getBooleanParamOrElse(appConfig, "output.compact", true)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   
	   //read input
	   val data = sparkCntxt.textFile(inputPath)
	   
	   //state transition with count 1
	   val laplaceCorrKeyedStatePair = sparkCntxt.parallelize(getLaplaceCorr(data, fieldDelimIn, keyFieldOrdinals, 
	       classAttrOrdinal, statesArr, mergeKeys))
	   val keyLen = if (mergeKeys) 3 else keyFieldOrdinals.length + 2
	   //laplaceCorrKeyedStatePair.foreach(r => r._1.check(keyLen))
       laplaceCorrKeyedStatePair.cache
       
       if (debugOn)
         print("state trans count")
	   var keyedStatePairs = if (seqLongFormat) {
	     keyedStatePairForLongFormat(data, fieldDelimIn, classAttrOrdinal, keyFieldOrdinals, seqFieldOrd,  stateFieldOrd, mergeKeys)
	   } else {
	     keyedStatePairForCompactFormat(data, fieldDelimIn, seqStartOrdinal,classAttrOrdinal, keyFieldOrdinals)
	   }.cache
	   //keyedStatePairs.foreach(r => r._1.check(keyLen))
	   
	   //laplace correction
       if (debugOn)
         print("laplace correction ")
	   keyedStatePairs = if (laplaceCorr) {
	     (keyedStatePairs ++ laplaceCorrKeyedStatePair).reduceByKey((v1, v2) => if (v1 > v2) v1 else v2)
	   } else {
	     keyedStatePairs
	   }
	   
	   //move state pairs from key to value
       if (debugOn)
         print("move state pair")
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
	   
	   val serTransRecs = if (outputCompact) {
	     //whole trans matrix in one record
	     transProb.map(r => {
	       r._1.toString() + fieldDelimOut + r._2.toString()
	     })
	   } else {
	     // trans matrix in multiple lines
	     transProb.flatMap(r => {
	       val recs = ArrayBuffer[String]()
	       recs += r._1.toString()
	       val trRecs = r._2.toString(false).split("\\n")
	       recs ++= trRecs
	       recs
	     })
	   }
	   
	   if (debugOn) {
	     val colTransProb = serTransRecs.collect
	     colTransProb.foreach(s => {
	       println("state trans probability:" + s)
	     })
	   }

	   if (saveOutput) {
	     serTransRecs.saveAsTextFile(outputPath)
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
		   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
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
    	 val keyRec =  getKey(classAttrOrdinal, items, keyFieldOrdinals)
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
  
   /**
   * @param config
   * @param paramName
   * @param defValue
   * @param errorMsg
   * @return
   */  
   def getKey(classAttrOrdinal:Option[Int], items:Array[String], keyFieldOrdinals:Array[Int]): Record =  {
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
   			val keyRec = Record(items, keyFieldOrdinals)
   			keyRec
   		}
	  }
	 keyRec
  }
  
  /**
  * @param config
  * @param paramName
  * @param errorMsg
  * @return
  */   
  def getLaplaceCorr(data:RDD[String], fieldDelimIn:String, keyFieldOrdinals:Array[Int], 
      classAttrOrdinal:Option[Int], statesArr:Array[String], mergeKeys:Boolean) :Array[(Record, Int)] = {
    val stateTrans = ArrayBuffer[(Record, Int)]()
    val keys = ArrayBuffer[Record]()
    if (mergeKeys) {
      val key = Record("all")
      keys += key
    } else {
      val uniqueKeys = data.map(line => {
    	 val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)   
    	 val key = getKey(classAttrOrdinal, items, keyFieldOrdinals)
    	 key
      }).distinct.collect
      keys ++= uniqueKeys
    }
    
    //laplace transition matrix
    val keyArr = keys.toArray
    keyArr.foreach(k => {
      //println("k " + k.toString())
      statesArr.foreach(s1 => {
        //println("s1 " + s1)
        statesArr.foreach(s2 => {
          //println("s2 " + s2)
          val trKey = Record(k.size + 2, k)
          trKey.add(s1, s2)
          println("count one state trans " + trKey.toString)
          val kv = (trKey, 1)
          stateTrans += kv
        })
      })
    })
    stateTrans.toArray
  }
  
}