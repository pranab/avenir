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

package org.avenir.spark.explore

import org.apache.spark.rdd.RDD
import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import org.chombo.spark.common.Record
import org.chombo.util.BasicUtils
import org.chombo.spark.common.GeneralUtility
import org.chombo.math.MathUtils
import org.avenir.util.PrincipalCompState

/**
* Online PCA with Spirit algorithm
* 
*/
object IncrementalPrincipalComponent extends JobConfiguration with GeneralUtility {
   /**
    * @param args
    * @return
    */
   def main(args: Array[String])  {
	   val appName = "incrementalPrincipalComponent"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //configurations
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val keyFieldOrdinals = toIntArray(getMandatoryIntListParam(appConfig, "id.field.ordinals"))
	   val quantFieldOrdinals = toIntArray(getMandatoryIntListParam(appConfig, "quant.field.ordinals"))
	   val seqFieldOrd = getMandatoryIntParam( appConfig, "seq.field.ordinal", 
	       "missing sequence field ordinal") 
	   val dimension = quantFieldOrdinals.length
	   
	   val stateFilePath = this.getOptionalStringParam(appConfig, "state.filePath")
	   val compState =  stateFilePath match {
	     case Some(path) => {
	       PrincipalCompState.load(path, fieldDelimOut).asScala.toMap
	     }
	     case None => {
	       Map[String,PrincipalCompState]()
	     }
	   }
	   val hiddenDimension =  this.getConditionalMandatoryIntParam(compState.isEmpty, appConfig, "hidden.dimension", 
	       "missing number of hidden units")
	   val precision = getIntParamOrElse(appConfig, "output.precision", 3)
	   val lambda = getDoubleParamOrElse(appConfig, "forget.factor", 0.96)
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)

	   val data = sparkCntxt.textFile(inputPath).cache
	   val modelData = data.map(line => {
		   val items = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		   val keyRec = Record(items, keyFieldOrdinals)
		   val valRec = Record(1 + quantFieldOrdinals.length)
		   valRec.addLong(items(seqFieldOrd).toLong)
		   Record.populateDoubleFields(items, quantFieldOrdinals, valRec)
		   (keyRec, valRec)
	   }).groupByKey.flatMap(r => {
	     val keyStr = r._1.toString
	     val values = r._2.toArray.sortWith((v1, v2) => v1.getLong(0) < v2.getLong(0))
	     val state = compState.get(keyStr) match {
	       case Some(state) => state
	       case None => new PrincipalCompState(keyStr, dimension, hiddenDimension)
	     }
	     val numHiddenStates = state.getNumHiddenStates()
	     val visibleEnergy = state.getVisibleEnergy()
	     val hiddenEnergy = state.getHiddenEnergy()
	     val hiddenUnitEnergy = state.getHiddenUnitEnergy()
	     var princComps = state.getPrincComps()
	     val pcMaRo = princComps.map(p => {
	       MathUtils.createRowMatrix(p)
	     })
	     val pcMaCo = princComps.map(p => {
	       MathUtils.createColMatrix(p)
	     })
	     
	     //all input
	     values.foreach(v => {
	       val vInp = v.getDoubleArray(1, v.size)
	       var inpMaRo = MathUtils.createRowMatrix(vInp)
	       var inpMaCo = MathUtils.createColMatrix(vInp)
	       //all hidden units
	       for (i <- 0 to numHiddenStates-1) {
	         //hidden component
	         val yMa = pcMaRo(i).times(inpMaCo)
	         val y = MathUtils.scalarFromMatrix(yMa)
	         
	         //update hidden unit energy
	         hiddenUnitEnergy(i) = lambda * hiddenUnitEnergy(i) + y * y
	         
	         //update pc
	         val proj  = pcMaCo(i).times(y)
	         val err = inpMaCo.minus(proj)
	         pcMaCo(i).plusEquals(err.times(y/hiddenUnitEnergy(i)))
	         pcMaRo(i) = pcMaCo(i).transpose()
	         
	         //update input
	         inpMaCo.minusEquals(pcMaCo(i).times(y))
	         inpMaRo = inpMaCo.transpose()
	       }
	       
	     })
	     princComps = pcMaRo.map(m => {
	       MathUtils.arrayFromRowMatrix(m)
	     })
	     
	     val energy = Array[Double](2)
	     energy(0) = visibleEnergy
	     energy(1) = hiddenEnergy
	     val prCompState = new PrincipalCompState(keyStr, dimension, numHiddenStates, energy, 
	         hiddenUnitEnergy, princComps)
	     prCompState.serialize(fieldDelimOut, precision).asScala
	   })
	   
	   if (debugOn) {
         val records = modelData.collect
         records.slice(0, 100).foreach(r => println(r))
       }
	   
	   if(saveOutput) {	   
	     modelData.saveAsTextFile(outputPath) 
	   }	 
	   
   }
   
   
}