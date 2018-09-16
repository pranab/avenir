# avenir-python: Machine Learning
# Author: Pranab Ghosh
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import print_function
import sys
from pyspark import SparkContext

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: wc.py <file> ")
		exit(-1)

	sc = SparkContext(appName="wordCount")
	lines = sc.textFile(sys.argv[1])
	words = lines.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)) 
	counts = words.reduceByKey(lambda a, b: a + b)
	cols = counts.collect()
	cols.foreach(print(_))
