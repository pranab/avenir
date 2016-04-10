#!/bin/bash
# contains everything needed to execute chombo in batch mode

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi
	
CHOMBO_JAR_NAME=/home/pranab/Projects/chombo/target/chombo-1.0.jar
AVENIR_JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
HDFS_BASE_DIR=/user/pranab/fit
PROP_FILE=/home/pranab/Projects/bin/avenir/fit.properties
HDFS_META_BASE_DIR=/user/pranab/meta

case "$1" in

"genInput")
 	./freq_items.py $2 $3 $4 > $5
	ls -l
	;;

"copyInput")
	hadoop fs -rm $HDFS_BASE_DIR/input/*
	hadoop fs -put $2 $HDFS_BASE_DIR/input
	hadoop fs -ls $HDFS_BASE_DIR/input
	;;

"tempFilter")
	echo "running mr TemporalFilter for temporal filtering"
	CLASS_NAME=org.chombo.mr.TemporalFilter
	IN_PATH=$HDFS_BASE_DIR/input
	OUT_PATH=$HDFS_BASE_DIR/filtered
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $CHOMBO_JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	hadoop fs -rmr $HDFS_BASE_DIR/filtered/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/filtered/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/filtered
	;;

"freqItems")
	echo "running mr FrequentItemsApriori for frequent items"
	CLASS_NAME=org.avenir.association.FrequentItemsApriori
	IN_PATH=$HDFS_BASE_DIR/filtered
	OUT_PATH=$HDFS_BASE_DIR/output_$2
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $AVENIR_JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	;;

*) 
	echo "unknown operation $1"
	;;

esac
