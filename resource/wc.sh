MASTER=spark://akash:7077
INPUT=file:///Users/pranab/Projects/bin/avenir/words.txt

$SPARK_HOME/bin/spark-submit --master $MASTER  wc.py $INPUT
