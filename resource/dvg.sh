#!/bin/bash
#converts high cardinality categorical aatributes to numerical

PROJECT_HOME=/Users/pranab/Projects
CHOMBO_JAR_NAME=$PROJECT_HOME/bin/chombo/uber-chombo-spark-1.0.jar
AVENIR_JAR_NAME=$PROJECT_HOME/bin/avenir/uber-avenir-spark-1.0.jar
MASTER=spark://akash.local:7077

case "$1" in

"genInput")	
	./sales_lead.py generate $2 > $3
	ls -l $3
	;;

"uniqueValues")	
	echo "running MultiArmBandit"
	CLASS_NAME=org.chombo.spark.explore.UniqueValueCounter
	INPUT=file:///Users/pranab/Projects/bin/avenir/input/dvg/leads.txt
	OUTPUT=file:///Users/pranab/Projects/bin/avenir/output/uvc
	rm -rf ./output/uvc
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
	--conf spark.ui.killEnabled=true --master $MASTER $CHOMBO_JAR_NAME  $INPUT $OUTPUT dvg.conf
	ls -l ./output/samp
	;;

"binValGen")	
	echo "running BinaryDummyVariableGenerator"
	CLASS_NAME=org.avenir.spark.util.BinaryDummyVariableGenerator
	INPUT=file:///Users/pranab/Projects/bin/avenir/input/dvg/leads.txt
	OUTPUT=file:///Users/pranab/Projects/bin/avenir/output/dvg
	rm -rf ./output/dvg
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
	--conf spark.ui.killEnabled=true --master $MASTER $AVENIR_JAR_NAME  $INPUT $OUTPUT dvg.conf
	ls -l ./output/samp
	;;

*) 
	echo "unknown operation $1"
	;;

esac
