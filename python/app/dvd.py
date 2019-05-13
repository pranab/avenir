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
	docSent = DocSentences(dataFile)
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
			
	# train
	sentAsToks = list(filter(lambda s: len(s) >= minSentenceLength, sentAsToks))
	if verbose:
		print "number of sentences after filter " + str(len(sentAsToks)) 
	return sentAsToks

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
			docSent = DocSentences(dataFile)
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
			
			# train
			sentAsToks = list(filter(lambda s: len(s) >= minSentenceLength, sentAsToks))
			print "number of sentences after filter " + str(len(sentAsToks)) 
			sentsAsText = list(map(lambda t: " ".join(t), sentAsToks))
			taggedSents = [TaggedDocument(s, [i]) for i, s in enumerate(sentsAsText)]
			d2v.train(taggedSents)
			print "**done with training"
	elif op == "genVec":
		if granularity == "document":
			print "document level vector generation"
		elif granularity == "sentence":
			print "sentence level vector generation"
			dataFile = config.getStringConfig("generate.data.file")[0]
			sentAsToks = getSentecesAsTokens(dataFile, minSentenceLength, verbose)
			vecEmbed = VectorEmbedding(dataFile, granularity)
			vecEmbed.getVectors(sentAsToks, d2v)
			if (config.getBooleanConfig("generate.vector.save")[0]):
				vecEmbed.save(config.getStringConfig("generate.vector.directory")[0],\
					config.getStringConfig("generate.vector.file")[0])

	else:
		print "invalid operation"
