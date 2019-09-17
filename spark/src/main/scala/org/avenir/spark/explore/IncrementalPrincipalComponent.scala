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
import org.apache.commons.lang3.ArrayUtils

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
	   val lowEnergyThreshold = getDoubleParamOrElse(appConfig, "energy.lowThreshold", 0.95)
	   val highEnergyThreshold = getDoubleParamOrElse(appConfig, "energy.highThreshold", 0.98)
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
	     var numHiddenStates = state.getNumHiddenStates()
	     var count = state.getCount()
	     var visibleEnergy = state.getVisibleEnergy()
	     var hiddenEnergy = state.getHiddenEnergy()
	     var hiddenUnitEnergy = state.getHiddenUnitEnergy()
	     var princComps = state.getPrincComps()
	     var pc = getPrinCompVectors(princComps)
	     var pcMaRo = pc._1
	     var pcMaCo = pc._2
	     val trInp = Array[Double](numHiddenStates)
	     
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
	         trInp(i) = y
	         
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
	         //check if number of hidden units need to change
	         visibleEnergy = ((count * visibleEnergy)  + MathUtils.getNorm(vInp)) / (count +1)
	         var totHiddenEnergy = 0.0
	         hiddenEnergy = hiddenEnergy.zip(trInp).map(r => {
	           ((count * r._1) + r._2 * r._2) / (count + 1)
	         })
	         hiddenEnergy.foreach(totHiddenEnergy += _ )
	         
	         if (totHiddenEnergy < lowEnergyThreshold * visibleEnergy) {
	           //add hidden unit
	           hiddenUnitEnergy = ArrayUtils.add(hiddenUnitEnergy, BasicUtils.sampleUniform(0, 1))
	           hiddenEnergy = ArrayUtils.add(hiddenEnergy, 0)
	           princComps = toPrincCompArray(pcMaRo)
	           princComps = extendPrinComp(princComps, dimension, numHiddenStates)
	           val pc = getPrinCompVectors(princComps)
	           pcMaRo = pc._1
	           pcMaCo = pc._2
	           numHiddenStates += 1
	         } else if (totHiddenEnergy > highEnergyThreshold * visibleEnergy) {
	           //remove last hidden unit
	           hiddenUnitEnergy = hiddenUnitEnergy.slice(0, numHiddenStates-2)
	           hiddenEnergy = hiddenEnergy.slice(0, numHiddenStates-2)
	           princComps = toPrincCompArray(pcMaRo)
	           princComps = princComps.slice(0, numHiddenStates-2)
	           val pc = getPrinCompVectors(princComps)
	           pcMaRo = pc._1
	           pcMaCo = pc._2
	           numHiddenStates -= 1
	         }
	         
	         count += 1
	     })
	     princComps = toPrincCompArray(pcMaRo)
	     
	     val energy = Array[Double](1 + numHiddenStates)
	     energy(0) = visibleEnergy
	     Array.copy(hiddenEnergy, 0, energy, 1, numHiddenStates)
	     
	     val prCompState = new PrincipalCompState(keyStr, dimension, numHiddenStates, count, energy, 
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
   
    /**
	* @param pcMaRo
    */   
    def toPrincCompArray(pcMaRo: Array[Jama.Matrix]) : Array[Array[Double]] =  {
      pcMaRo.map(m => MathUtils.arrayFromRowMatrix(m))
    }
   
    /**
	* @param princComps
    */   
    def getPrinCompVectors(princComps:Array[Array[Double]]): (Array[Jama.Matrix], Array[Jama.Matrix]) = {
      val pcMaRo = princComps.map(p => {
       MathUtils.createRowMatrix(p)
      })
      val pcMaCo = princComps.map(p => {
       MathUtils.createColMatrix(p)
      })
	  (pcMaRo, pcMaCo)   
    }
   
    /**
	* @param princComps
    */   
    def extendPrinComp(princComps:Array[Array[Double]], dimension:Int, numHiddenStates:Int) :Array[Array[Double]] =  {
      val newprincComps = Array.ofDim[Array[Double]](princComps.length+1)
      for (i <- 0 to princComps.length) {
        newprincComps(i) = princComps(i)
      }
      newprincComps(princComps.length) = BasicUtils.createOneHotDoubleArray(dimension, numHiddenStates)
      newprincComps
    }
    
}