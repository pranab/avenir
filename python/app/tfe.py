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

def buildNGram(docClean, nGram):
	"""
	build ngram
	"""
	for words in docClean:
		nGram.countDocNGrams(words)
	#print("vocab size before {}".format(biGram.getVocabSize()))
	nGram.remLowCount(3)
	#print("vocab size after {}".format(biGram.getVocabSize()))
	nGram.getNGramFreq()

def vectorise(docClean, nGram):
	"""
	vectorise
	"""
	for words in docClean:
		vec = nGram.getVector(words, True, True)
		if (nGram.getNonZeroCount() > 0):
			vec = toStrList(vec, 6)
			print(",".join(vec))
	

if __name__ == "__main__":
	configFile  = sys.argv[1]
	mode = sys.argv[2] 
	ae = AutoEncoder(configFile)
	config = ae.getConfig()
	verbose = config.getBooleanConfig("common.verbose")[0]
	
	if mode == "distr" or mode == "vectorise":
		dirPath = sys.argv[3] 
		ngramType = sys.argv[4] 
		preprocessor = TextPreProcessor()
		# document list
		if os.path.isdir(dirPath):
			#one doc per file
			docComplete, filePaths  = getFileContent(dirPath, verbose)
		else:
			#one doc per line in a file
			docComplete = getFileLines(dirPath)
				
		# pre process
		docClean = [clean(doc, preprocessor, verbose) for doc in docComplete]
		
		# convert to bigrams
		if ngramType == "bi":
			nGram = BiGram(None, verbose)
		elif ngramType == "tri":
			nGram = TriGram(None, verbose)
		else:
			raise ValueError("invalid ngram type")	
			
		for words in docClean:
			nGram.countDocNGrams(words)
		#print("vocab size before {}".format(biGram.getVocabSize()))
		nGram.remLowCount(3)
		#print("vocab size after {}".format(biGram.getVocabSize()))
		nGram.getNGramFreq()

		# word counts as list
		nGram.getNGramIndex(True)
		
		# vectorise
		if mode == "vectorise":
			for words in docClean:
				vec = nGram.getVector(words, True, True)
				if (nGram.getNonZeroCount() > 0):
					vec = toStrList(vec, 6)
					print(",".join(vec))
	
	elif mode == "train":
		ae.buildModel()
		ae.trainAe()

	elif mode == "params":
		ae.buildModel()
		ae.getParams()

	elif mode == "numEncode":
		ngramFilePath = sys.argv[3] 
		textFilePath = sys.argv[3] 
		
		#load ngram
		nGram = NGram.load(ngramFilengramFilePath)
		
		# pre process and vectorise
		docComplete = getFileLines(textFilePath)
		docClean = [clean(doc, preprocessor, verbose) for doc in docComplete]
		vectorise(docClean, nGram)		
		
	elif mode == "encode":
		ae.buildModel()
		ae.encode()
		
	else:
		raise ValueError("invalid command")
		
			
