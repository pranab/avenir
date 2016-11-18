#!/bin/bash
# contains everything needed to execute chombo in batch mode

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi
	
JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
HDFS_BASE_DIR=/user/pranab
PROP_FILE=/home/pranab/Projects/bin/avenir/detr.properties
HDFS_META_BASE_DIR=/user/pranab/meta

case "$1" in

"genData")
	 ./call_hangup.py  $2 > $3
	 ls -l $3
;;

"loadInput")
	hadoop fs -rm $HDFS_BASE_DIR/detr/input/$2
	hadoop fs -put $2 $HDFS_BASE_DIR/detr/input
	hadoop fs -ls $HDFS_BASE_DIR/detr/input
;;

"loadMeta")
	hadoop fs -rm $HDFS_META_BASE_DIR/detr/$2
	hadoop fs -put $2 $HDFS_META_BASE_DIR/detr
	hadoop fs -ls $HDFS_META_BASE_DIR/detr
;;

"mvDecPathFile")
	hadoop fs -mv /user/pranab/detr/other/decPathOut.txt /user/pranab/detr/other/decPathIn.txt
;;

"decTree")
	echo "running MR DecisionTreeBuilder"
	CLASS_NAME=org.avenir.tree.DecisionTreeBuilder
	IN_PATH=$HDFS_BASE_DIR/detr/input
	OUT_PATH=$HDFS_BASE_DIR/detr/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	hadoop fs -ls $HDFS_BASE_DIR/detr/output
;;

*) 
	echo "unknown operation $1"
	;;

esac

