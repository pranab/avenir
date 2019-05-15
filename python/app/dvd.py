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
from dv import *

# get sentences as token list
def getSentecesAsTokens(dataFile, minSentenceLength, verbose):
	docSent = DocSentences(dataFile, verbose)
	sents = docSent.getSentences()
	if verbose:
		print "*******sentences as text"
		for s in sents:
			print s

	sentAsToks = docSent.getSentencesAsTokens()
	
	if verbose:
		print "*******sentences as token list"
		for t in sentAsToks:
			print len(t)
			
	sentAsToksIndx = [(i,st) for i,st in enumerate(sentAsToks)]
	sentAsToksIndx = list(filter(lambda s: len(s[1]) >= minSentenceLength, sentAsToksIndx))
	sentAsToks = list(map(lambda sti: sti[1], sentAsToksIndx))	
	sentIndexes = list(map(lambda sti: sti[0], sentAsToksIndx))	

	print "number of sentences after filter " + str(len(sentAsToks)) 
	#print sentIndexes
	return (sentAsToks, sentIndexes)

if __name__ == "__main__":
	configFile  = sys.argv[1]
	op  = sys.argv[2]
	d2v = DocToVec(configFile)

	# execute		
	config = d2v.getConfig()
	verbose = config.getBooleanConfig("common.verbose")[0]
	granularity = config.getStringConfig("train.text.granularity")[0]
	minSentenceLength = config.getIntConfig("train.min.sentence.length")[0]
	preprocessor = TextPreProcessor()

	if op == "train":
		if granularity == "document":
			print "document level training"
		elif granularity == "sentence":
			print "sentence level training"
			dataFile = config.getStringConfig("train.data.file")[0]
			sentAsToks = getSentecesAsTokens(dataFile, minSentenceLength, verbose)[0]
			print "number of sentences after filter " + str(len(sentAsToks)) 
			sentsAsText = list(map(lambda t: " ".join(t), sentAsToks))
			
			# train
			taggedSents = [TaggedDocument(s, [i]) for i, s in enumerate(sentsAsText)]
			d2v.train(taggedSents)
			print "**done with training"
	
	elif op == "genVec":
		if granularity == "document":
			print "document level vector generation"
		elif granularity == "sentence":
			print "sentence level vector generation"
			dataFile = config.getStringConfig("generate.data.file")[0]
			sentAsToks = getSentecesAsTokens(dataFile, minSentenceLength, verbose)[0]
			vecEmbed = VectorEmbedding(dataFile, granularity)
			vecEmbed.getVectors(sentAsToks, d2v)
			if (config.getBooleanConfig("generate.vector.save")[0]):
				vecEmbed.save(config.getStringConfig("generate.vector.directory")[0],\
					config.getStringConfig("generate.vector.file")[0])

	elif op == "neighbor":
		if granularity == "document":
			print "document level nearest neighbors"
		elif granularity == "sentence":
			print "sentence level nearest neighbors"
			sentIndx  = int(sys.argv[3])
			vecEmbed = VectorEmbedding.load(config.getStringConfig("generate.vector.directory")[0],\
				config.getStringConfig("generate.vector.file")[0])
			distances = vecEmbed.getDistances(sentIndx, config.getStringConfig("distance.algorithm")[0],\
				config.getFloatConfig("distance.algorithm.param")[0])
			print distances

	elif op == "show":
		if granularity == "document":
			print "show document"
		elif granularity == "sentence":
			print "show sentence"
			sentIndx  = int(sys.argv[3])
			dataFile = config.getStringConfig("generate.data.file")[0]
			sentAsToks, sentIndexes = getSentecesAsTokens(dataFile, minSentenceLength, verbose)
			sentIndex = sentIndexes[sentIndx]
			print "real sentence index " + str(sentIndex)
			print getSentences(dataFile)[sentIndex]

	else:
		print "invalid operation"
