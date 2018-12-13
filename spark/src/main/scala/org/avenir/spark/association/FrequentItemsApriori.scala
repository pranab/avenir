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

package org.avenir.spark.association

import org.chombo.spark.common.JobConfiguration
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import com.typesafe.config.Config
import org.apache.spark.rdd.RDD
import org.chombo.util.BasicUtils
import org.chombo.spark.common.Record
import scala.collection.mutable.ArrayBuffer
import org.chombo.spark.common.GeneralUtility

/**
* @param keyFields
* @return
*/
object FrequentItemsApriori extends JobConfiguration with GeneralUtility {
  
   /**
    * @param args
    * @return
    */
   def main(args: Array[String]) {
	   val appName = "contTimeStateTransitionStats"
	   val Array(inputPath: String, outputPath: String, configFile: String) = getCommandLineArgs(args, 3)
	   val config = createConfig(configFile)
	   val sparkConf = createSparkConf(appName, config, false)
	   val sparkCntxt = new SparkContext(sparkConf)
	   val appConfig = config.getConfig(appName)
	   
	   //config params
	   val fieldDelimIn = getStringParamOrElse(appConfig, "field.delim.in", ",")
	   val fieldDelimOut = getStringParamOrElse(appConfig, "field.delim.out", ",")
	   val useTransId = getBooleanParamOrElse(appConfig, "use.transId", true)
	   val tranIdOrd = getOptionalIntParam(appConfig, "tran.idOrd")
	   val withTransId = tranIdOrd match {
	       case Some(ord) => true
		   case None => false
	   }
	   val keyFields = getOptionalIntListParam(appConfig, "id.fieldOrdinals")
	   val keyFieldOrdinals = toOptionalIntegerArray(keyFields)
	   val maxItemSetLength = getOptionalIntParam(appConfig, "max.itemSetLength")
	   val minSupport = getMandatoryDoubleParam(appConfig, "min.support", "missing minimumm support")
	   val itemFiledOrdinals = toIntegerArray(getMandatoryIntListParam(appConfig, "item.filedOrdinals", "missing items field ordinals"))
	   val debugOn = getBooleanParamOrElse(appConfig, "debug.on", false)
	   val saveOutput = getBooleanParamOrElse(appConfig, "save.output", true)
	   val baseKeyLen = getOptinalArrayLength(keyFieldOrdinals, 1)
	   
	   //input
	   val data = sparkCntxt.textFile(inputPath).cache
	   val counts = data.map(line => {
		   val fields = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		   val key = Record(baseKeyLen)
		   populateFields(fields, fromOptionalIntegerToIntArray(keyFieldOrdinals), key, "all")
	       (key, 1)
	   }).reduceByKey((v1, v2) => v1 + v2).collectAsMap.mapValues(v => v.toDouble)

	   var done = false
	   var itemSetLen = 1
	   val itemsSets : Option[RDD[(Record, Record)]] = None
	   var itemsSetsMap : Option[Map[Record, Map[Record, Record]]] = None
	   
	   while(!done) {
	     val data = sparkCntxt.textFile(inputPath)
	     val keyLen = baseKeyLen + itemSetLen
	     
	     val nextItemSets = itemsSetsMap match {
	       //items sets exits
	       case Some(itemsSetsMap) => {
	          var itemSets = data.flatMap(line => {
	            val records = ArrayBuffer[(Record, Record)]()
	            val fields = BasicUtils.getTrimmedFields(line, fieldDelimIn)
	            
	            //base key
	            val baseKey = keyFieldOrdinals match {
	              case Some(ordinals) => Record(fields, ordinals)
	              case None => Record("all")
	            }
	            
	            //items set in current record
	            val thisItemSet = Record(fields, itemFiledOrdinals)
	            
	            //existing frequent items sets
	            val itemSetsWithTransId = itemsSetsMap.get(baseKey) match {
	              case Some(itemSets) => itemSets
	              case None => throw new IllegalStateException("")
	            }
	            
	            tranIdOrd match {
	                 //trans Id based
		    	     case Some(ord) => {
		    	       val transId = fields(ord)
		    	       processNextItemSet(fields, itemSetsWithTransId,thisItemSet, baseKey, withTransId, ord) match {
		    	         case Some(itemSetRec) => records += itemSetRec
		    	         case None => 
		    	       }
		    	     }
		    	     
		    	     //without trans Id
		    	     case None => {
		    	       processNextItemSet(fields, itemSetsWithTransId,thisItemSet, baseKey, withTransId) match {
		    	         case Some(itemSetRec) => records += itemSetRec
		    	         case None => 
		    	       }
		    	     }
		    	}
	            records
	          })
	          
	          //reduce to remove duplicates
	          itemSets = itemSets.reduceByKey((v1, v2) => {
	            if (withTransId) v1 else Record(1, v1.getInt(0) + v2.getInt(0))
	          })
	          
	         //only item sets with support over threshold
		     val colItemSets = filterForSupport(itemSets, baseKeyLen, counts, withTransId, minSupport).collect
	         
	         //convert to a nested map to be used in the next iteration
		     createItemSetMap(colItemSets, baseKeyLen)
	       }
	        
	       //first time
	       case None => {
		     val itemSets = data.flatMap(line => {
		    	 val fields = BasicUtils.getTrimmedFields(line, fieldDelimIn)
		         itemFiledOrdinals.map(i => {
		    	   val key = Record(keyLen)
		    	   populateFields(fields, fromOptionalIntegerToIntArray(keyFieldOrdinals), key, "all")
		    	   key.addString(fields(i))
		          
		    	   val value = tranIdOrd match {
		    	     case Some(ord) => Record(fields(ord))
		    	     case None => Record(1, 1)
		    	   }
		    	   (key, value)
		         })
		     }).reduceByKey((v1, v2) => if (withTransId) Record(v1, v2) else Record(1, v1.getInt(0) + v2.getInt(0)))
		     
		     //only item sets with support over threshold
		     val colItemSets = filterForSupport(itemSets, baseKeyLen, counts, withTransId, minSupport).collect
		     
		     //convert to a nested map to be used in the next iteration
		     createItemSetMap(colItemSets, baseKeyLen)
	       }
	     }
	   
	     //itemsSetsMap = Some(nextItemSets.collectAsMap)
	     itemSetLen += 1
	     itemsSetsMap = Some(nextItemSets)
	   }

   }
   
  /**
  * @param rec
  * @param baseKeyLen
  * @return
  */
  def getComponents(rec:(Record, Record), baseKeyLen : Int) : (Record, Record, Record) = {
     val baseKey = Record(rec._1, 0, baseKeyLen)
	 val itemsSet = Record(rec._1, baseKeyLen, rec._1.size)
	 val value = rec._2
	 (baseKey, itemsSet, value)
   }

  /**
  * @param colItemSets
  * @param baseKeyLen
  * @return
  */
  def createItemSetMap(colItemSets: Array[(Record, Record)], baseKeyLen:Int) : Map[Record, Map[Record, Record]] = {
     var itemSetsByKey = Map[Record, Map[Record, Record]]()
     colItemSets.foreach(r => {
      val key = r._1
      val baseKey = Record(key, 0, baseKeyLen)
      val itemsSet = Record(key, baseKeyLen, key.size)
      val value = r._2
      
      var itemSetData = itemSetsByKey.getOrElse(baseKey, Map[Record, Record]())
      if(itemSetData.isEmpty) itemSetsByKey += (baseKey -> itemSetData)
      itemSetData += (itemsSet -> value)
     })
    itemSetsByKey
  }
  
  /**
  * @param fields
  * @param itemSetsWithTransId
  * @param thisItemSet
  * @param baseKey
  * @param withTransId
  * @param transIdOrd
  * @return
  */
  def processNextItemSet(fields: Array[String], itemSetsWithTransId: Map[Record,Record],
      thisItemSet:Record, baseKey:Record, withTransId:Boolean, transIdOrd:Int = 0) : Option[(Record, Record)] = {
	   var RecWithMoreItem : Option[(Record, Record)] = None
	   val transId = if (withTransId) fields(transIdOrd) else ""
	   
	   itemSetsWithTransId.foreach(r => {
	     val itemsSet = r._1
	     val transIds = r._2
	     
	     //only if current freq items contains this transaction or current items set is contained with all items in record
	     if (withTransId && transIds.findString(transId)  || !withTransId && thisItemSet.containsAllStrings(itemsSet)) {
	       thisItemSet.initialize()
	       while (thisItemSet.hasNext()) {
	         val thisItem = thisItemSet.getString()
	         
	         //current items set does not contain this item
	         if (!itemsSet.findString(thisItem)) {
	           //create item set record with additional item
	           val newItemsSet = Record(itemsSet.size + 1, itemsSet)
	           newItemsSet.add(thisItem)
	           newItemsSet.sort()
	           val newKey = Record(baseKey, newItemsSet)
	           if (!withTransId) {
	             transIds.addInt(0,1)
	           }
	           val rec = (newKey, transIds)
	           RecWithMoreItem = Some(rec)
	         }
	       }
	     }
	   }) 
	   RecWithMoreItem
  }
  
  /**
  * @param itemSets
  * @param baseKeyLen
  * @param counts
  * @param withTransId
  * @param minSupport
  * @return
  */
  def filterForSupport(itemSets: RDD[(Record, Record)], baseKeyLen:Int, counts: scala.collection.Map[Record,Double],
     withTransId:Boolean, minSupport:Double) : RDD[(Record, Record)] = {
     itemSets.filter(r => {
       val baseKey = Record(r._1, 0, baseKeyLen)
       val totCount = counts.get(baseKey) match {
         case Some (count) => count
         case None => throw new IllegalStateException("missing total count")
       }
       val count = if (withTransId) r._2.size else  r._2.getInt(0)
       val sup = count / totCount
       sup > minSupport
     })
  }
  
}