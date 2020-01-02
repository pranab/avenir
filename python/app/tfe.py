#!/usr/bin/python

#!/Users/pranab/Tools/anaconda/bin/python

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

import os
import sys
from random import randint
from random import uniform
import time
import uuid
import nltk
from nltk.corpus import movie_reviews
from random import shuffle
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
sys.path.append(os.path.abspath("../unsupv"))
from util import *
from sampler import *
from ae import *
from preprocess import *


if __name__ == "__main__":
	configFile  = sys.argv[1]
	mode = sys.argv[2] 
	ae = AutoEncoder(configFile)
	config = ae.getConfig()
	verbose = config.getBooleanConfig("common.verbose")[0]
	
	if mode == "distr" or mode == "vectorise":
		preprocessor = TextPreProcessor()
		# document list
		dirPath = sys.argv[2] 
		if os.path.isdir(dirPath):
			#one doc per file
			docComplete, filePaths  = getFileContent(dirPath, verbose)
		else:
			#one doc per line in a file
			docComplete = getFileLines(dirPath)
				
		# pre process
		docClean = [clean(doc, preprocessor, verbose) for doc in docComplete]
		
		# convert to bigrams
		biGram = BiGram(None, verbose)
		for words in docClean:
			biGram.countDocNGrams(words)
		#print("vocab size before {}".format(biGram.getVocabSize()))
		biGram.remLowCount(3)
		#print("vocab size after {}".format(biGram.getVocabSize()))
		biGram.getNGramFreq()

		# vectorise
		if mode == "vectorise":
			for words in docClean:
				vec = biGram.getVector(words, True, True)
				if (biGram.getNonZeroCount() > 0):
					vec = toStrList(vec, 6)
					print(",".join(vec))
	
	elif mode == "train":
		ae.buildModel()
		ae.train()
		
			
