#!/bin/bash

PROJECT_HOME=/Users/pranab/Projects
JAR_NAME=$PROJECT_HOME/bin/avenir/uber-avenir-spark-1.0.jar
CLASS_NAME=org.avenir.spark.markov.StateTransitionRate
MASTER=spark://Pranab-Ghoshs-MacBook-Pro.local:7077
EXTRA_CLASSPATH=$PROJECT_HOME/chombo/target/chombo-1.0.jar,$PROJECT_HOME/chombo/spark/target/scala-2.10/chombo-spark_2.10-1.0.jar

$SPARK_HOME/bin/spark-submit --class $CLASS_NAME   \
--conf spark.ui.killEnabled=true --master $MASTER $JAR_NAME  file:///Users/pranab/Projects/bin/avenir/atm_trans.txt \
file:///Users/pranab/Projects/bin/avenir/str atmTrans.conf

