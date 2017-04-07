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
import java.io.FileInputStream

/**
 * Predicts based on Markov chain classifier
 * @param args
 * @return
 */
object  MarkovModelClassifier  extends JobConfiguration {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "markovModelClassifier"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyFieldOrdinals = getMandatoryIntListParam(appConfig, "id.field.ordinals").asScala.toArray
	   val classAttrOrdinal = getOptionalIntParam(appConfig, "class.attr.ordinal")
	   val seqStartOrdinal = getMandatoryIntParam(appConfig, "seq.start.ordinal")
	   val states = getMandatoryStringListParam(appConfig, "state.list", "missing state list")
	   val statesArr = BasicUtils.fromListToStringArray(states)
	   val scale = getMandatoryIntParam(appConfig, "trans.prob.scale")
	   //val outputPrecision = getIntParamOrElse(appConfig, "output.precision", 3);
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   val prMatFilePath = getMandatoryStringParam(appConfig, "prob.matrix.file.path", "missing probability matrix file path")
	   val classValues = getMandatoryStringListParam(appConfig, "class.values", "missing class attribute values").asScala.toArray
	   val logOddsThreshold = getMandatoryDoubleParam(appConfig, "log.odds.threshold", 
	       "missing log odds threshold")
	   val validationMode = classAttrOrdinal match {
		     case Some(classOrd:Int) => true		     
		     case None => false		     
		   }
	   
	   //state transition prob matrix
	   val keyedStateTransProb = buildStateTransProbMatrix(prMatFilePath, keyFieldOrdinals.length, fieldDelimIn, 
			   statesArr, scale)
	   
	   //read input
	   val data = sparkCntxt.textFile(inputPath)
	   
	   val predResults = data.map(line => {
		   val items = line.split(fieldDelimIn, -1)

		   val keyRec = classAttrOrdinal match {
		     case Some(classOrd:Int) => {
		       //validation
		       val classVal = items(classOrd)
		       val keyRec = Record(keyFieldOrdinals.length + 2, items, keyFieldOrdinals)
			   keyRec.addString(classVal)
		       keyRec
		     }
		     
		     case None => {
		       //prediction
		       val keyRec = Record(keyFieldOrdinals.length + 1, items, keyFieldOrdinals)
		       keyRec
		     }
		   }

		   //positive class value state transition prob matrix
		   val posClassKeyRec = Record(keyFieldOrdinals.length + 1, items, keyFieldOrdinals)
		   posClassKeyRec.addString(classValues(0))
		   val posClassStateTransProb = keyedStateTransProb.get(posClassKeyRec) match {
		     case Some(s : StateTransitionProbability) => s
		     case None => throw new IllegalStateException("missing state transition probability matrix for positive class value")
		   }
		   
		   //negative class value state transition prob matrix
		   val negClassKeyRec = Record(keyFieldOrdinals.length + 1, items, keyFieldOrdinals)
		   negClassKeyRec.addString(classValues(1))
		   val negClassStateTransProb = keyedStateTransProb.get(negClassKeyRec) match {
		     case Some(s : StateTransitionProbability) => s
		     case None => throw new IllegalStateException("missing state transition probability matrix for negative class value")
		   }
		   
		   //log odds
		   var logOdds = 0.0
		   for (idx <- seqStartOrdinal+1 to items.length -1) {
		     val frState = items(idx - 1)
		     val toState = items(idx)
		     logOdds += Math.log(posClassStateTransProb.get(frState, toState) / 
	         				negClassStateTransProb.get(frState, toState));
		   }
		   
		   //predict
		   val predClassVal = if (logOdds > logOddsThreshold) 
		     classValues(0)
		    else 
		     classValues(1)
			   
		   keyRec.addString(predClassVal)
		   keyRec   
	   })
	   
	   if (debugOn) {
	     val colPredResults = predResults.collect
	     colPredResults.foreach(r => {
	       println(r)
	     })
	   }
	   
	   if (saveOutput) {
	     predResults.saveAsTextFile(outputPath)
	   }
	   
   }
   
   /**
    * @param prMatFilePath
    * @param keyLen
    * @param fieldDelimIn
    * @param statesArr
    * @param scale
    * @return
    */
   def buildStateTransProbMatrix(prMatFilePath : String, keyLen : Int, fieldDelimIn: String, 
       statesArr : Array[String], scale : Int) : Map[Record, StateTransitionProbability] = {
     var keyedStateTransProb = Map[Record, StateTransitionProbability]()
     val inpStr = new FileInputStream(prMatFilePath)
     val lines = BasicUtils.getFileLines(inpStr).asScala.toList
     
     lines.map(line => {
    	 val items = line.split(fieldDelimIn, -1)
    	 
    	 val rec = Record(keyLen + 2, items, 0, keyLen)
    	 rec.addString(items(keyLen))
    	 
    	 val stTransProb = new StateTransitionProbability(statesArr, statesArr)
    	 stTransProb.withScale(scale)
    	 val size = statesArr.length
    	 var offset = keyLen + 1
    	 for (row  <- 0 to size-1) {
    	   stTransProb.deseralizeRow(items, offset, row)
    	   offset += size
    	 }
    	 keyedStateTransProb += (rec -> stTransProb)
    	 keyedStateTransProb
     })
     
     keyedStateTransProb
   }
   
}