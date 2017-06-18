#!/bin/bash

PROJECT_HOME=/Users/pranab/Projects
JAR_NAME=$PROJECT_HOME/bin/avenir/uber-avenir-spark-1.0.jar
MASTER=spark://akash:7077

case "$1" in

"simulatedAnnealing")
	echo "running simulatedAnnealing"
	CLASS_NAME=org.avenir.spark.optimize.SimulatedAnnealing
	INPUT=file:///Users/pranab/Projects/bin/avenir/input/opt/sia/soln.txt
	OUTPUT=file:///Users/pranab/Projects/bin/avenir/output/opt/sia
	rm -rf ./output/opt/sia
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
	--conf spark.ui.killEnabled=true --master $MASTER $JAR_NAME  $OUTPUT opt.conf
;;



*) 
	echo "unknown operation $1"
	;;
esac