#!/bin/bash

PROJECT_HOME=/Users/pranab/Projects
JAR_NAME=$PROJECT_HOME/avenir/spark/target/scala-2.10/avenir-spark_2.10-1.0.jar
CLASS_NAME=org.avenir.spark.markov.StateTransitionRate
MASTER=spark://Pranab-Ghoshs-MacBook-Pro.local:7077
EXTRA_CLASSPATH=$PROJECT_HOME/chombo/target/chombo-1.0.jar,$PROJECT_HOME/chombo/spark/target/scala-2.10/chombo-spark_2.10-1.0.jar

$SPARK_HOME/bin/spark-submit --class $CLASS_NAME  --driver-class-path $EXTRA_CLASSPATH \
--conf spark.ui.killEnabled=true $JAR_NAME $MASTER atmTrans.txt str.txt atmTrans.conf
