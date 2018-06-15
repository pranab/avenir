#!/bin/bash
#converts high cardinality categorical aatributes to numerical

PROJECT_HOME=/Users/pranab/Projects
CHOMBO_JAR_NAME=$PROJECT_HOME/bin/chombo/uber-avenir-spark-1.0.jar
AVENIR_JAR_NAME=$PROJECT_HOME/bin/avenir/uber-avenir-spark-1.0.jar
MASTER=spark://akash:7077

case "$1" in

"genInput")	
	./loan_approve.py generate $2 > $3
	cp $3 ./input/caen/
	ls -l $3
	;;

"rmStatFile")
	rm -rf ./other/caen
	;;

"encodeLoo")	
	echo "running CategoricalLeaveOneOutEncoding"
	CLASS_NAME=org.avenir.spark.explore.CategoricalLeaveOneOutEncoding
	INPUT=file:///Users/pranab/Projects/bin/avenir/input/caen/loan.txt
	OUTPUT=file:///Users/pranab/Projects/bin/avenir/output/caen
	rm -rf ./output/caen
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
	--conf spark.ui.killEnabled=true --master $MASTER $AVENIR_JAR_NAME  $INPUT $OUTPUT caen.conf
	ls -l ./output/caen
	;;

*) 
	echo "unknown operation $1"
	;;

esac
