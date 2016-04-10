#!/bin/bash
# contains everything needed to execute sifarish in batch mode

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi
	
JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
HDFS_BASE_DIR=/user/pranab/conv
PROP_FILE=/home/pranab/Projects/bin/avenir/conv.properties
HDFS_META_BASE_DIR=/user/pranab/meta/conv

case "$1" in

"genTrainData")  
	./visit_history.py $2 $3 label > $4
    ;;

"expTrainData")  
	hadoop fs -rm $HDFS_BASE_DIR/train/input/*
	hadoop fs -put $2 $HDFS_BASE_DIR/train/input
	hadoop fs -ls $HDFS_BASE_DIR/train/input
    ;;

"trainConv")  
	echo "running mr"
	CLASS_NAME=org.avenir.markov.MarkovStateTransitionModel
	IN_PATH=$HDFS_BASE_DIR/train/input
	OUT_PATH=$HDFS_BASE_DIR/train/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $JAR_NAME $CLASS_NAME -Dconf.path=$PROP_FILE $IN_PATH $OUT_PATH
	hadoop fs -rmr $HDFS_BASE_DIR/train/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/train/output/_SUCCESS	
	hadoop fs -ls $HDFS_BASE_DIR/train/output
    ;;

"genTestData")  
	./visit_history.py $2 $3  > $4
    ;;

"expTestData")  
	hadoop fs -rm $HDFS_BASE_DIR/pred/input/*
	hadoop fs -put $2 $HDFS_BASE_DIR/pred/input
	hadoop fs -ls $HDFS_BASE_DIR/pred/input
    ;;

"expModelData")  
	hadoop fs -cp $HDFS_BASE_DIR/train/output/part-r-00000 $HDFS_META_BASE_DIR/mcc_conv.txt
	hadoop fs -ls $HDFS_META_BASE_DIR
    ;;

"predConv")  
	echo "running mr"
	CLASS_NAME=org.avenir.markov.MarkovModelClassifier
	IN_PATH=$HDFS_BASE_DIR/pred/input
	OUT_PATH=$HDFS_BASE_DIR/pred/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $JAR_NAME $CLASS_NAME -Dconf.path=$PROP_FILE $IN_PATH $OUT_PATH
	hadoop fs -rmr $HDFS_BASE_DIR/pred/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/pred/output/_SUCCESS	
	hadoop fs -ls $HDFS_BASE_DIR/pred/output
    ;;

*) 
	echo "unknown operation $1"
	;;

esac
