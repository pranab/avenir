#!/bin/bash

if [ $# -lt 1 ]
then
        echo "Usage : $0 operation"
        exit
fi
	
HDFS_BASE_DIR=/user/pranab/knn
HDFS_META_BASE_DIR=/user/pranab/meta/knn
PROP_FILE=/home/pranab/Projects/bin/avenir/knn.properties


case "$1" in

"createTrainingData")
	# usage: ./knn.sh createTrainingData <num_of users>  <training_data_file> 
	echo "create training data for eLearning signal"
	./elearn.py $2 > $3 
    ;;

"createTestData")
	# usage: ./knn.sh createTrainingData <num_of users> <test_data_file> 
	echo "create training/test data for eLearning signal"
	./elearn.py $2 > $3
    ;;
    
"expData")  
	# usage: ./knn.sh expData <training_data_file> <test_data_file>
	echo "exporting input data to HDFS"
	hadoop fs -rmr $HDFS_BASE_DIR/inp/*
	hadoop fs -put $2 $HDFS_BASE_DIR/inp
	hadoop fs -put $3 $HDFS_BASE_DIR/inp
	hadoop fs -ls $HDFS_BASE_DIR/inp
    ;;

"expSchema")  
	echo "exporting entity schema to HDFS"
	hadoop fs -rmr $HDFS_META_BASE_DIR/$2
	hadoop fs -put $2 $HDFS_META_BASE_DIR
	hadoop fs -ls $HDFS_META_BASE_DIR
    ;;

"computeDistance")
	echo "running distance(similarity) calculating MR"
	JAR_NAME=/home/pranab/Projects/sifarish/target/sifarish-1.0.jar
	CLASS_NAME=org.sifarish.feature.SameTypeSimilarity
	IN_PATH=$HDFS_BASE_DIR/inp
	OUT_PATH=$HDFS_BASE_DIR/simi
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	hadoop fs -rmr $HDFS_BASE_DIR/simi/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/simi/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/simi
	;;

"expTrData")  
	# usage: ./knn.sh expData <training_data_file> 
	echo "xxx exporting input data to HDFS"
	hadoop fs -rmr $HDFS_BASE_DIR/binp/*
	hadoop fs -put $2 $HDFS_BASE_DIR/binp
	hadoop fs -ls $HDFS_BASE_DIR/binp
    ;;

"bayesianDistr")
	echo "running feature and class probability distribution calculating MR"
	JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
	CLASS_NAME=org.avenir.bayesian.BayesianDistribution
	IN_PATH=$HDFS_BASE_DIR/inp/$2
	OUT_PATH=$HDFS_BASE_DIR/distr
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	hadoop fs -rmr $HDFS_BASE_DIR/distr/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/distr/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/distr
	;;

"bayesianPredictor")
	echo "running feature posterior probability distribution calculating MR"
	JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
	CLASS_NAME=org.avenir.bayesian.BayesianPredictor
	IN_PATH=$HDFS_BASE_DIR/inp/$2
	OUT_PATH=$HDFS_BASE_DIR/pprob
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	hadoop fs -rmr $HDFS_BASE_DIR/pprob/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/pprob/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/pprob
	;;

"renameProbDistrFile")	
	echo "renaming probability distribution file"
	hadoop fs -mv $HDFS_BASE_DIR/pprob/$2 $HDFS_BASE_DIR/pprob/$3  
	hadoop fs -ls $HDFS_BASE_DIR/pprob
	;;

"joinFeatureDistr")
	echo "running feature posterior probability distribution and distance joiner MR"
	JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
	CLASS_NAME=org.avenir.knn.FeatureCondProbJoiner
	IN_PATH=$HDFS_BASE_DIR/simi,$HDFS_BASE_DIR/pprob
	OUT_PATH=$HDFS_BASE_DIR/join
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	hadoop fs -rmr $HDFS_BASE_DIR/join/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/join/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/join
	;;

"knnClassifier")
	# 2nd arg is join if you are using class condioned weighting, simi otherwise
	echo "running KNN classifier MR"
	JAR_NAME=/home/pranab/Projects/avenir/target/avenir-1.0.jar
	CLASS_NAME=org.avenir.knn.NearestNeighbor
	IN_PATH=$HDFS_BASE_DIR/$2
	OUT_PATH=$HDFS_BASE_DIR/output
	echo "input $IN_PATH output $OUT_PATH"
	hadoop fs -rmr $OUT_PATH
	echo "removed output dir"
	hadoop jar $JAR_NAME  $CLASS_NAME -Dconf.path=$PROP_FILE  $IN_PATH  $OUT_PATH
	hadoop fs -rmr $HDFS_BASE_DIR/output/_logs
	hadoop fs -rmr $HDFS_BASE_DIR/output/_SUCCESS
	hadoop fs -ls $HDFS_BASE_DIR/output
	;;

*) 
	echo "unknown operation $1"
	;;

esac
