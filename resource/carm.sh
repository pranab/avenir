#!/bin/bash
# contains everything needed to execute sifarish in batch mode

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi
	
JAR_NAME=/Users/pranab/Projects/bin/avenir/uber-avenir-1.0.jar
HDFS_BASE_DIR=/projects/carm
PROP_FILE=/Users/pranab/Projects/bin/avenir/carm.properties
HDFS_META_BASE_DIR=/projects/carm/meta

case "$1" in

"mutInfo")  
	echo "running MR  MutualInformation"
	CLASS_NAME=org.avenir.explore.MutualInformation
	IN_PATH=$HDFS_BASE_DIR/input
	OUT_PATH=$HDFS_BASE_DIR/output
	echo "input $IN_PATH output $OUT_PATH"
	hdfs dfs -rm -r $OUT_PATH
	echo "removed output dir $OUT_PATH"
	yarn jar $JAR_NAME $CLASS_NAME -Dconf.path=$PROP_FILE $IN_PATH $OUT_PATH
	hdfs dfs -rm -r $HDFS_BASE_DIR/output/_logs
	hdfs dfs -rm -r $HDFS_BASE_DIR/output/_SUCCESS	
	hdfs dfs -ls $HDFS_BASE_DIR/output
    ;;


*) 
	echo "unknown operation $1"
	;;

esac
