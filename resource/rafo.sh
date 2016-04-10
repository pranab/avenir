#!/bin/bash
# contains everything needed to execute random forest

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi

JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
HDFS_BASE_DIR=/user/pranab/raf
HDFS_META_DIR=/user/pranab/meta/raf
PROP_FILE=/home/pranab/Projects/bin/avenir/rafo.properties

case "$1" in

"genInput")
 	./telecom_churn.py $2 > $3
	ls -l
	;;

"copyInput")
	hadoop fs -rm $HDFS_BASE_DIR/input/*
	hadoop fs -put $2 $HDFS_BASE_DIR/input
	hadoop fs -ls $HDFS_BASE_DIR/input
	;;

"copySchema")
	hadoop fs -rm $HDFS_META_DIR/*
	hadoop fs -put $2 $HDFS_META_DIR
	hadoop fs -ls $HDFS_META_DIR
	;;

"decTree")
	CLASS_NAME=org.avenir.tree.DecisionTreeBuilder
	echo "running MR DecisionTreeBuilder"
	IN_PATH=$HDFS_BASE_DIR/input
	OUT_PATH=$HDFS_BASE_DIR/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	;;

*) 
	echo "unknown operation $1"
	;;

esac
