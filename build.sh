#!/bin/bash

check()
{
  if [ "$?" != "0" ]; then
    echo "**failed!" 1>&2
    exit 1
  fi
}
echo "building hadoop..."
mvn clean install
check 
sbt publishLocal
check

echo "building spark..."
cd spark
sbt clean package
check
sbt publishLocal
check
cd ..
