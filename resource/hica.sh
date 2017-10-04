#!/bin/bash
#converts high cardinality categorical aatributes to numerical

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi

JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
CHOMBO_JAR_NAME=/home/pranab/Projects/chombo/target/chombo-1.0.jar
HDFS_BASE_DIR=/user/pranab/hica
HDFS_META_DIR=/user/pranab/meta/hica
PROP_FILE=/home/pranab/Projects/bin/avenir/hica.properties


case "$1" in

"genInput")
 	./lead_time.py $2 > $3
	ls -l
	;;

"copyInput")
	hadoop fs -rm $HDFS_BASE_DIR/cace/input/*
	hadoop fs -put $2 $HDFS_BASE_DIR/cace/input
	hadoop fs -ls $HDFS_BASE_DIR/cace/input
	;;

"copySchema")
	hadoop fs -rm $HDFS_META_DIR/$2
	hadoop fs -put $2 $HDFS_META_DIR
	hadoop fs -ls $HDFS_META_DIR
	;;

"hiCard")	
	CLASS_NAME=org.avenir.explore.CategoricalContinuousEncoding
	echo "running mr $CLASS_NAME"
	IN_PATH=$HDFS_BASE_DIR/cace/input
	OUT_PATH=$HDFS_BASE_DIR/cace/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir $OUT_PATH"
	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH	
	hadoop fs -rmr $HDFS_BASE_DIR/cace/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/cace/output/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/cace/output
	;;
	
"transform")
	CLASS_NAME=org.chombo.mr.Transformer
	echo "running mr $CLASS_NAME"
	IN_PATH=$HDFS_BASE_DIR/cace/input
	OUT_PATH=$HDFS_BASE_DIR/trans/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $CHOMBO_JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	hadoop fs -rmr $HDFS_BASE_DIR/trans/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/trans/output/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/trans/output
	;;

*) 
	echo "unknown operation $1"
	;;

esac
	

