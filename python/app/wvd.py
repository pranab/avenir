#!/usr/bin/python
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

# Package imports
import os
import sys
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import jprops
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from preprocess import *
from util import *
from wv import *


if __name__ == "__main__":
	configFile  = sys.argv[1]
	op  = sys.argv[2]
	w2v = WordToVec(configFile)

	# execute		
	config = w2v.getConfig()
	verbose = config.getBooleanConfig("common.verbose")[0]
	preprocessor = TextPreProcessor()

	if op == "train":
		path = config.getStringConfig("train.data.dir")[0]
		docComplete, filePaths  = getFileContent(path, verbose)

		# pre process
		docClean = [clean(doc, preprocessor, verbose) for doc in docComplete]

		#train
		w2v.train(docClean)

	elif op == "fsw":
		result = w2v.findSimilarWords(["blockchain"], 5)
		print result


