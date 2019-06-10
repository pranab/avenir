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


# top elements by odds ration
def topByOddsRatio(distr, oddsRatio, minCount=None):
	s = 0.0
	sel = []
	for d in distr:
		s += d[1]
		sel.append(d)
		if (s / (1.0 - s)) > oddsRatio:
			break	
	if minCount and len(sel) < minCount:
		sel = distr[:minCount]
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
	dtMinCount = config.getIntConfig("analyze.doc.topic.min.count")[0]
	twMinCount = config.getIntConfig("analyze.topic.word.min.count")[0]

	docTopics = []
	docByTopic = {}
	wordsByTopic = {}
	
	if verbose:
		print "result size " + str(len(result))

	# all docs 
	for didx, dt in enumerate(result):
		print "\n** next doc " + filePaths[didx]
		docResult = {}
		dt.sort(key=takeSecond, reverse=True)
		print "doc topic distribution " + str(dt)
		dtTop = topByOddsRatio(dt, dtOddsRatio, dtMinCount)
		docTopics.append(dtTop)
		print "filtered doc topic distribution " + str(dtTop)
		# all topics
		for t in dtTop:
			print "next topic with distr : " + str(t)
			tid = t[0]
			appendKeyedList(docByTopic, tid, didx)
			tw = lda.getTopicTerms(tid, twTopMax)
			twTop = topByOddsRatio(tw, twOddsRatio, twMinCount)
			print "filtered topic word distribution " + str(twTop)
			docResult[tid] = (t[1], twTop)
			if wordsByTopic.get(tid) is None:
				for w, p in twTop:
					appendKeyedList(wordsByTopic, tid, w)
			if verbose:
				print "topic words: " + str(twTop)
		
		# net word dist for doc
		wdList = netWordDistr(docResult, verbose)
		print "final doc word distr " + str(wdList)
	return (docTopics, docByTopic, wordsByTopic)

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

def getAllTopics(config, lda):
	numTopic = config.getIntConfig("train.num.topics")[0]
	topics = []
	for i in range(numTopic):
		topTerms = lda.getTopicTerms(i, 10)
		topics.append(topTerms)
	return topics

#######################

configFile  = sys.argv[1]
lda = LatentDirichletAllocation(configFile)

# execute	
mode = sys.argv[2] if len(sys.argv) == 3 else lda.getMode()
config = lda.getConfig()
verbose = lda.getConfig().getBooleanConfig("common.verbose")[0]

# processor object
preprocessor = TextPreProcessor()

print "running mode: " + mode
if mode == "train":
	# dcument list
	path = config.getStringConfig("train.data.dir")[0]
	docComplete, filePaths  = getFileContent(path, verbose)

	# pre process
	docClean = [clean(doc, preprocessor, verbose) for doc in docComplete]

	# train
	result = lda.train(docClean)

	if lda.isSingleNumOfTopics():
		# one topic number
		if verbose:
			print "one topic number"
		result = lda.getDocTopics()		
		docTopics, docByTopic, wordsByTopic = processResult(config, result, lda, filePaths, verbose)
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
	path = config.getStringConfig("analyze.data.dir")[0]
	docComplete, filePaths  = getFileContent(path, verbose)

	# pre process
	docClean = [clean(doc, preprocessor, verbose) for doc in docComplete]

	# analyze all docs
	result = lda.analyze(docClean)
	docTopics, docByTopic, wordsByTopic = processResult(config, result, lda, filePaths, verbose)

	#doc topics with file name
	fileDocTopics = zip(filePaths, docTopics)
	print "**file topics"
	print fileDocTopics

	# each topic
	ceFilt = config.getBooleanConfig("analyze.word.cross.entropy.filter")[0]
	if ceFilt:
		saveFile = config.getStringConfig("analyze.base.word.distr.file")[0]
		glTf = loadTermDistr(saveFile).getWordFreq()
		print "filtering topic words for each topic with cross entropy"
		topicWords = []
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

			#final topic words
			tWords = list(map(lambda wc: wc[0], wordCe))
			t = (tid, tWords)
			topicWords.append(t)
		
		#word list for each topic
		topicWords.sort(key=takeFirst)
		print "**filtered topic words"
		print topicWords

elif mode == "buildBaseTermDistr":
	categories = ['rec.autos', 'soc.religion.christian','comp.graphics', 'sci.med', 'talk.politics.misc',\
		'talk.religion.misc', 'sci.med']
	twentyTrain = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
	print "pre processing " + str(len(twentyTrain.data)) + " docs"
	docsClean = [clean(doc, preprocessor, verbose) for doc in twentyTrain.data]	

	saveFile = config.getStringConfig("analyze.base.word.distr.file")[0]
	crateTermDistr(docsClean, None, saveFile)

elif mode == "loadBaseTermDistr":
	saveFile = config.getStringConfig("analyze.base.word.distr.file")[0]
	loadTermDistr(saveFile)

elif mode == "getAllTopics":
	#allTopics = lda.getAllTopicTerms()	
	topTerms = getAllTopics(config, lda)
	print topTerms

