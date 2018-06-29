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

def getFileContent(dirPathParam):
	# dcument list
	docComplete  = []
	path = config.getStringConfig(dirPathParam)[0]
	filePaths = getAllFiles(path)

	# read files
	for filePath in filePaths:
		print "next file " + filePath
		with open(filePath, 'r') as contentFile:
			content = contentFile.read()
			docComplete.append(content)
	return docComplete

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

# soring
def takeSecond(elem):
    return elem[1]

# 
def topByOddsRatio(distr, oddsRatio):
	s = 0.0
	sel = []
	for d in distr:
		s += d[1]
		sel.append(d)
		if (s / (1.0 - s)) > oddsRatio:
			break	
	return sel


#######################
configFile  = sys.argv[1]
lda = LatentDirichletAllocation(configFile)

# execute		
mode = lda.getMode()
config = lda.getConfig()

# processor object
preprocessor = TextPreProcessor()

print "running mode: " + mode
if mode == "train":
	# dcument list
	docComplete  = getFileContent("train.data.dir")

	# pre process
	docClean = [clean(doc, preprocessor) for doc in docComplete]

	# train
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

elif mode == "analyze":
	# dcument list
	docComplete  = getFileContent("analyze.data.dir")

	# pre process
	docClean = [clean(doc, preprocessor) for doc in docComplete]
	result = lda.analyze(docClean)
	dtOddsRatio = config.getFloatConfig("analyze.doc.topic.odds.ratio")[0]
	twOddsRatio = config.getFloatConfig("analyze.topic.word.odds.ratio")[0]
	for dt in result:
		dt.sort(key=takeSecond, reverse=True)
		dtTop = topByOddsRatio(dt, dtOddsRatio)
		for t in dtTop:
			print 
			print "topic: " + str(t)
			tid = t[0]
			tw = lda.getTopicTerms(tid, 50)
			twTop = topByOddsRatio(tw, twOddsRatio)
			print "words: " + str(twTop)


