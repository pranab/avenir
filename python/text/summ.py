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
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
import jprops
import math
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from util import *
from mlutil import *
from preprocess import *
nltk.download('punkt')
	
# text summarizer
class SummariseByTermFreq:
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["summ.min.sentence.length"] = (5, None)
		defValues["summ.size"] = (5, None)
		defValues["summ.byCount"] = (True, None)
		defValues["summ.length.normalizer"] = ("linear", None)
		defValues["summ.show.score"] = (False, None)
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]

	# get config object
	def getConfig(self):
		return self.config
			
	def getSummary(self, filePath):
		minSentLength = self.config.getIntConfig("summ.min.sentence.length")[0]
		normalizer = self.config.getStringConfig("summ.length.normalizer")[0]
		
		docSent = DocSentences(filePath, minSentLength, self.verbose)
		sents = docSent.getSentences()

		termTable = TfIdf(None, False)
		if self.verbose:
			print "*******sentences as text"
			for s in sents:
				print s

		# term count table for all words
		sentWords = docSent.getSentencesAsTokens()
		for seWords in sentWords:
			termTable.countDocWords(seWords)
		
		#score for each sentence
		sentScores = []
		for seWords in sentWords:
			counts = list(map(lambda w: termTable.getCount(w), seWords))	
			totCount = reduce((lambda c1, c2: c1 + c2), counts)
			sentScores.append(totCount)
		
		#sentence length	
		sentLens = list(map(lambda seWords: len(seWords), sentWords))	
		minLen = min(sentLens)
			
 		#sentence index, score, sentence length
 		zippedSents = zip(range(len(sents)), sentScores, sentLens)
 		
 		#normalize scores
 		if normalizer == "linear":
 			zippedSents = list(map(lambda zs: (zs[0], zs[1]/zs[2]), zippedSents))
 		if normalizer == "log":
 			zippedSents = list(map(lambda zs: (zs[0], int(zs[1] * (1.0 / (1.0 + math.log(zs[2]/minLen))))), zippedSents))
 		
 		# sort of decreasing score	
 		sortedSents = sorted(zippedSents, key=takeSecond, reverse=True)
 		print "after soerting num sentences " + str(len(sortedSents))
 		
 		#retain top sentences
 		summSize  = self.config.getIntConfig("summ.size")[0]
 		byCount = self.config.getBooleanConfig("summ.byCount")[0]
 		if not byCount:
 			summSize = (len(sortedSents) * summSize) / 100
 		print "summSize " + str(summSize)
 		topSents = sortedSents[:summSize]
 		
 		#sort sentence by position 
 		topSents = sorted(topSents, key=takeFirst)	
 		return list(map(lambda ts: (sents[ts[0]], ts[1]), topSents))
 		
 		