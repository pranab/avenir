#!/bin/bash
# contains everything needed to execute random forest

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi

JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
HDFS_BASE_DIR=/user/pranab/ovsa
HDFS_META_DIR=/user/pranab/meta/ovsa
PROP_FILE=/home/pranab/Projects/bin/avenir/ovsa.properties
CHOMBO_JAR_NAME=/home/pranab/Projects/chombo/target/chombo-1.0.jar


case "$1" in

"genInput")
 	./machine_op.py $2 > $3
	ls -l
	;;

"copyInput")
	hadoop fs -rm $HDFS_BASE_DIR/norm/input/*
	hadoop fs -put $2 $HDFS_BASE_DIR/norm/input
	hadoop fs -ls $HDFS_BASE_DIR/norm/input
	;;

"copySchema")
	hadoop fs -rm $HDFS_META_DIR/$2
	hadoop fs -put $2 $HDFS_META_DIR
	hadoop fs -ls $HDFS_META_DIR
	;;

"normalize")	
	CLASS_NAME=org.chombo.mr.Normalizer
	echo "running mr $CLASS_NAME"
	IN_PATH=$HDFS_BASE_DIR/norm/input
	OUT_PATH=$HDFS_BASE_DIR/norm/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir $OUT_PATH"

	hadoop jar $CHOMBO_JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH	
	hadoop fs -rmr $HDFS_BASE_DIR/norm/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/norm/output/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/norm/output
	;;
	
"filtMaj")	
	CLASS_NAME=org.chombo.mr.Projection
	echo "running mr $CLASS_NAME"
	IN_PATH=$HDFS_BASE_DIR/norm/output
	OUT_PATH=$HDFS_BASE_DIR/filt/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir $OUT_PATH"

	hadoop jar $CHOMBO_JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH	
	hadoop fs -rmr $HDFS_BASE_DIR/filt/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/filt/output/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/filt/output
	;;

"distance")	
	CLASS_NAME=org.chombo.mr.RecordSimilarity
	echo "running mr $CLASS_NAME"
	IN_PATH=$HDFS_BASE_DIR/filt/output
	OUT_PATH=$HDFS_BASE_DIR/dist/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir $OUT_PATH"

	hadoop jar $CHOMBO_JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH	
	hadoop fs -rmr $HDFS_BASE_DIR/dist/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/dist/output/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/dist/output
	;;
	
"topMatches")	
	CLASS_NAME=org.avenir.explore.TopMatchesByClass
	echo "running mr $CLASS_NAME"
	IN_PATH=$HDFS_BASE_DIR/dist/output
	OUT_PATH=$HDFS_BASE_DIR/topn/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir $OUT_PATH"

	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH	
	hadoop fs -rmr $HDFS_BASE_DIR/topn/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/topn/output/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/topn/output
	;;
	
"ovSamp")	
	CLASS_NAME=org.avenir.explore.ClassBasedOverSampler
	echo "running mr $CLASS_NAME"
	IN_PATH=$HDFS_BASE_DIR/topn/output
	OUT_PATH=$HDFS_BASE_DIR/cbos/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir $OUT_PATH"

	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH	
	hadoop fs -rmr $HDFS_BASE_DIR/cbos/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/cbos/output/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/cbos/output
	;;

*) 
	echo "unknown operation $1"
	;;

esac
	

