#!/usr/bin/python

import os
import sys
import nltk
from nltk.corpus import movie_reviews
from random import shuffle
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import jprops
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from preprocess import *
from util import *
from mlutil import *
from lda import *


def clean(doc, preprocessor):
	#print "--raw doc"
	#print doc
	words = preprocessor.tokenize(doc)
	words = preprocessor.toLowercase(words)
	words = preprocessor.removeStopwords(words)
	words = preprocessor.removePunctuation(words)
	words = preprocessor.lemmatizeWords(words)
	#print "--after pre processing"
	#print words
	return words

#######################
configFile  = sys.argv[1]
lda = LatentDirichletAllocation(configFile)

# execute		
mode = lda.getMode()
config = lda.getConfig()
print "running mode: " + mode
if mode == "train":
	# dcument list
	docComplete  = []
	path = config.getStringConfig("train.data.dir")[0]
	filePaths = getAllFiles(path)
	config = lda.getConfig()

	# read files
	for filePath in filePaths:
		print "next file " + filePath
		with open(filePath, 'r') as contentFile:
			content = contentFile.read()
			docComplete.append(content)
	

	# processor object
	preprocessor = TextPreProcessor()

	# pre process
	docClean = [clean(doc, preprocessor) for doc in docComplete]

	result = lda.train(docClean)

	plotPerplexity = lda.getConfig().getBooleanConfig("train.plot.perplexity")[0]
	if plotPerplexity:
		# plot
		plt.plot(result[0], result[1])
 
		# naming the x and y axis
		plt.xlabel("num of topics")
		plt.ylabel("peplexity")
 
		# giving a title to my graph
		plt.title("perplexity variation")
		plt.show()




