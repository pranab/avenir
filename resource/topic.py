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
from sklearn.datasets import fetch_20newsgroups
import pickle
import math
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from preprocess import *
from util import *
from mlutil import *
from lda import *

def getFileContent(dirPathParam, verbose):
	# dcument list
	docComplete  = []
	path = config.getStringConfig(dirPathParam)[0]
	filePaths = getAllFiles(path)

	# read files
	for filePath in filePaths:
		if verbose:
			print "next file " + filePath
		with open(filePath, 'r') as contentFile:
			content = contentFile.read()
			docComplete.append(content)
	return (docComplete, filePaths)

def clean(doc, preprocessor, verbose):
	if verbose:
		print "--raw doc"
		print doc
	words = preprocessor.tokenize(doc)
	words = preprocessor.toLowercase(words)
	words = preprocessor.removeStopwords(words)
	words = preprocessor.removePunctuation(words)
	words = preprocessor.lemmatizeWords(words)
	if verbose:
		print "--after pre processing"
		print words
	return words

# soring
def takeSecond(elem):
    return elem[1]

# top elements by odds ration
def topByOddsRatio(distr, oddsRatio):
	s = 0.0
	sel = []
	for d in distr:
		s += d[1]
		sel.append(d)
		if (s / (1.0 - s)) > oddsRatio:
			break	
	return sel

# document word distr marginalizing over topic
def netWordDistr(docResult, verbore):
	wordDistr = {}
	for tid, (tp,tw) in docResult.iteritems():
		if verbose:
			print "topic id " + str(tid)
			print "topic pr " + str(tp)
			print "word distr " + str(tw)
		
		for w, wp in tw:
			p = wp * tp
			addToKeyedCounter(wordDistr, w, p)
	
	wdList = [(k, v) for k, v in wordDistr.items()]	
	wdList.sort(key=takeSecond, reverse=True)
	return wdList

# analyzes results
def processResult(config, result, lda, filePaths, verbose):
	dtOddsRatio = config.getFloatConfig("analyze.doc.topic.odds.ratio")[0]
	twOddsRatio = config.getFloatConfig("analyze.topic.word.odds.ratio")[0]
	twTopMax = config.getIntConfig("analyze.topic.word.top.max")[0]
	docByTopic = {}
	wordsByTopic = {}
	
	if verbose:
		print "result size " + str(len(result))

	# all docs 
	for didx, dt in enumerate(result):
		print "doc " + filePaths[didx]
		docResult = {}
		dt.sort(key=takeSecond, reverse=True)
		dtTop = topByOddsRatio(dt, dtOddsRatio)
		# all topics
		for t in dtTop:
			if verbose:
				print "topic and distr : " + str(t)
			tid = t[0]
			appendKeyedList(docByTopic, tid, didx)
			tw = lda.getTopicTerms(tid, twTopMax)
			twTop = topByOddsRatio(tw, twOddsRatio)
			docResult[tid] = (t[1], twTop)
			if wordsByTopic.get(tid) is None:
				for w, p in twTop:
					appendKeyedList(wordsByTopic, tid, w)
			if verbose:
				print "topic words: " + str(twTop)
		
		# net word dist for doc
		wdList = netWordDistr(docResult, verbose)
		print "final doc words distr " + str(wdList)
	return (docByTopic, wordsByTopic)

# base term ditsribution
def crateTermDistr(docs, vocFilt, saveFile):
	print "term distribution"
	print "num of docs " + str(len(docs))
	tfidf = TfIdf(vocFilt, False)
	for d in docs:
		tfidf.countDocWords(d)
	tf = tfidf.getWordFreq()
	print "term distr size " + str(len(tf))
	verbose = False
	if verbose:
		for t in tf.items():
			print "%s	%.6f" %(t[0], t[1])
	
	if saveFile is not None:
		tfidf.save(saveFile)
	return tf

# load term distr from file
def loadTermDistr(saveFile):
	tfidf = TfIdf.load(saveFile)
	return tfidf

#######################
configFile  = sys.argv[1]
lda = LatentDirichletAllocation(configFile)

# execute		
mode = lda.getMode()
config = lda.getConfig()
verbose = lda.getConfig().getBooleanConfig("common.verbose")[0]

# processor object
preprocessor = TextPreProcessor()

if verbose:
	print "running mode: " + mode
if mode == "train":
	# dcument list
	docComplete, filePaths  = getFileContent("train.data.dir", verbose)

	# pre process
	docClean = [clean(doc, preprocessor, verbose) for doc in docComplete]

	# train
	result = lda.train(docClean)

	if lda.isSingleNumOfTopics():
		# one topic number
		if verbose:
			print "one topic number"
		result = lda.getDocTopics()		
		docByTopic, wordsByTopic = processResult(config, result, lda, filePaths. verbose)
	else:
		# multiple topic numbers to decide optimum number
		if verbose:
			print "multiple topic numbers"
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
	docComplete, filePaths  = getFileContent("analyze.data.dir", verbose)

	# pre process
	docClean = [clean(doc, preprocessor, verbose) for doc in docComplete]

	# analyze all docs
	result = lda.analyze(docClean)
	docByTopic, wordsByTopic = processResult(config, result, lda, filePaths, verbose)


	# each topic
	ceFilt = config.getBooleanConfig("analyze.word.cross.entropy.filter")[0]
	if ceFilt:
		saveFile = config.getStringConfig("analyze.base.word.distr.file")[0]
		glTf = loadTermDistr(saveFile).getWordFreq()
		for tid in docByTopic.keys():
			print "topic id " + str(tid)
			# all docs for the topic
			dids = docByTopic.get(tid)
			docs = [docClean[did] for did in dids]
			print "num of docs " + str(len(docs))
			tf = crateTermDistr(docs, None, None)
			
			wordCe = []
			skippedWords = []
			for w in wordsByTopic.get(tid):
				p = tf.get(w)
				q = glTf.get(w)
				if not (p is None or q is None):
					ce = p * math.log(p / q)
					wordCe.append((w, ce))
				else:
					skippedWords.append(w)
			wordCe.sort(key=takeSecond, reverse=True)			
			print "words after cross entropy filter" + str(wordCe)	
			print "skipped " + str(skippedWords) 

elif mode == "buildBaseTermDistr":
	categories = ['rec.autos', 'soc.religion.christian','comp.graphics', 'sci.med', 'talk.politics.misc']
	twentyTrain = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
	print "pre processing " + str(len(twentyTrain.data)) + " docs"
	docsClean = [clean(doc, preprocessor, verbose) for doc in twentyTrain.data]	

	saveFile = config.getStringConfig("analyze.base.word.distr.file")[0]
	crateTermDistr(docsClean, None, saveFile)

elif mode == "loadBaseTermDistr":
	saveFile = config.getStringConfig("analyze.base.word.distr.file")[0]
	loadTermDistr(saveFile)


