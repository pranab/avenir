#!/bin/bash

PROJECT_HOME=/Users/pranab/Projects
CHOMBO_JAR_NAME=$PROJECT_HOME/bin/chombo/uber-chombo-spark-1.0.jar
AVENIR_JAR_NAME=$PROJECT_HOME/bin/avenir/uber-avenir-spark-1.0.jar
MASTER=spark://akash.lan:7077

case "$1" in

"cpInp")
	echo "copying input files"
	rm $PROJECT_HOME/bin/avenir/input/ks/intv/*
	cp $2 $PROJECT_HOME/bin/avenir/input/ks/intv/
	ls -l $PROJECT_HOME/bin/avenir/input/ks/intv/
;;

"timeIntv")
	echo "running TimeIntervalGenerator"
	CLASS_NAME=org.chombo.spark.explore.TimeIntervalGenerator
	INPUT=file:///Users/pranab/Projects/bin/avenir/input/ks/intv/*
	OUTPUT=file:///Users/pranab/Projects/bin/avenir/output/ks/intv
	rm -rf ./output/ks/intv
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
	--conf spark.ui.killEnabled=true --master $MASTER $CHOMBO_JAR_NAME  $INPUT $OUTPUT ks.conf
	rm -rf ./output/ks/intv/_SUCCESS
;;

"numDistrStat")
	echo "running NumericalAttrDistrStats Spark job"
	CLASS_NAME=org.chombo.spark.explore.NumericalAttrDistrStats
	INPUT=file:///Users/pranab/Projects/bin/avenir/output/ks/intv/*
	OUTPUT=file:///Users/pranab/Projects/bin/avenir/output/ks/nds
	rm -rf ./output/ks/nds
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
	--conf spark.ui.killEnabled=true --master $MASTER $CHOMBO_JAR_NAME  $INPUT $OUTPUT ks.conf
	rm -rf ./output/ks/nds/_SUCCESS
;;

"rmModel")
	echo "removing model files"
	rm $PROJECT_HOME/bin/avenir/input/ks/stat/nds.txt
	ls -l $PROJECT_HOME/bin/avenir/input/ks/stat/
;;

"cpModel")
	echo "copying model files"
	NDS_FILES=$PROJECT_HOME/bin/avenir/output/ks/nds/*
	NDS_DIR=$PROJECT_HOME/bin/avenir/input/ks/stat
	for f in $NDS_FILES
	do
  		echo "Copying file $f ..."
  		cat $f >> $NDS_DIR/$2
	done
    ls -l $NDS_DIR/
;;

"ksStat")
	echo "running KolmogorovSmirnovModelDrift Spark job"
	CLASS_NAME=org.avenir.spark.explore.KolmogorovSmirnovModelDrift
	INPUT=file:///Users/pranab/Projects/bin/avenir/input/ks/stat/*
	OUTPUT=file:///Users/pranab/Projects/bin/avenir/output/ks/stat
	rm -rf ./output/ks/stat
	$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
	--conf spark.ui.killEnabled=true --master $MASTER $AVENIR_JAR_NAME  $INPUT $OUTPUT ks.conf
	rm -rf ./output/ks/stat/_SUCCESS
;;


*) 
	echo "unknown operation $1"
	;;

esac