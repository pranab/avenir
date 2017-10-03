#!/bin/bash
#converts high cardinality categorical aatributes to numerical

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi

JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
HDFS_BASE_DIR=/user/pranab/hica
PROP_FILE=/home/pranab/Projects/bin/avenir/hica.properties


case "$1" in

"genInput")
 	./lead_time.py $2 > $3
	ls -l
	;;

"copyInput")
	hadoop fs -rm $HDFS_BASE_DIR/input/*
	hadoop fs -put $2 $HDFS_BASE_DIR/input
	hadoop fs -ls $HDFS_BASE_DIR/input
	;;

"hiCard")	
	CLASS_NAME=org.avenir.explore.CategoricalContinuousEncoding
	echo "running mr $CLASS_NAME"
	IN_PATH=$HDFS_BASE_DIR/input
	OUT_PATH=$HDFS_BASE_DIR/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir $OUT_PATH"

	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH	
	hadoop fs -rmr $HDFS_BASE_DIR/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/output/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/output
	;;

*) 
	echo "unknown operation $1"
	;;

esac
	

