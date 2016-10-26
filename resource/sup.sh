#!/bin/bash

PROJECT_HOME=/Users/pranab/Projects
JAR_NAME=$PROJECT_HOME/bin/avenir/uber-avenir-spark-1.0.jar
MASTER=spark://akash:7077

case "$1" in

"transRate")
	echo "running transRate"
	CLASS_NAME=org.avenir.spark.markov.StateTransitionRate
	INPUT=file:///Users/pranab/Projects/bin/avenir/input/sup/fulfill.txt
	OUTPUT=file:///Users/pranab/Projects/bin/avenir/output/sup/tra
	rm -rf ./output/sup/tra
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
	--conf spark.ui.killEnabled=true --master $MASTER $JAR_NAME  $INPUT $OUTPUT sup.conf
;;

"rateStat")
	echo "running rateStat"
	CLASS_NAME=org.avenir.spark.markov.ContTimeStateTransitionStats
	INPUT=file:///Users/pranab/Projects/bin/avenir/input/sup/fulfill_states.txt
	OUTPUT=file:///Users/pranab/Projects/bin/avenir/output/sup/ras
	rm -rf ./output/sup/ras
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
	--conf spark.ui.killEnabled=true --master $MASTER $JAR_NAME  $INPUT $OUTPUT sup.conf
;;

*) 
	echo "unknown operation $1"
	;;

esac
