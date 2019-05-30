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
from gensim.summarization.summarizer import summarize
from gensim import corpora, models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from util import *
from mlutil import *
from preprocess import *
nltk.download('punkt')
	
# text summarizer based on term frequency
class TermFreqSumm:
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["common.min.sentence.length"] = (5, None)
		defValues["common.size"] = (5, None)
		defValues["common.byCount"] = (True, None)
		defValues["summ.length.normalizer"] = ("linear", None)
		defValues["common.show.score"] = (False, None)
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]

	# get config object
	def getConfig(self):
		return self.config
	
	# get summary sentences		
	def getSummary(self, filePath, text=None):
		minSentLength = self.config.getIntConfig("common.min.sentence.length")[0]
		normalizer = self.config.getStringConfig("summ.length.normalizer")[0]
		
		docSent = DocSentences(filePath, minSentLength, self.verbose, text)
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
 		summSize  = self.config.getIntConfig("common.size")[0]
 		byCount = self.config.getBooleanConfig("common.byCount")[0]
 		if not byCount:
 			summSize = (len(sortedSents) * summSize) / 100
 		print "summSize " + str(summSize)
 		topSents = sortedSents[:summSize]
 		
 		#sort sentence by position 
 		topSents = sorted(topSents, key=takeFirst)	
 		return list(map(lambda ts: (sents[ts[0]], ts[1]), topSents))
 		
# sum basic summarizer		
class SumBasicSumm:
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["common.min.sentence.length"] = (5, None)
		defValues["Common.size"] = (5, None)
		defValues["common.byCount"] = (True, None)
		defValues["common.show.score"] = (False, None)
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
 	
	# get config object
	def getConfig(self):
		return self.config
			
	def getSummary(self, filePath, text=None):
		minSentLength = self.config.getIntConfig("common.min.sentence.length")[0]
		docSent = DocSentences(filePath, minSentLength, self.verbose, text)
		sents = docSent.getSentences()
		
		# term count table for all words
		termTable = TfIdf(None, False)
		sentWords = docSent.getSentencesAsTokens()
		for seWords in sentWords:
			termTable.countDocWords(seWords)
		wordFreq = termTable.getWordFreq()
 		
		summSize  = self.config.getIntConfig("common.size")[0]
		byCount = self.config.getBooleanConfig("common.byCount")[0]
		if not byCount:
 			summSize = (len(sortedSents) * summSize) / 100

		topSentences = []
		for i in range(summSize):
			#score for each sentence
			sentScores = []
			for seWords in sentWords:
				frquencies = list(map(lambda w: wordFreq[w], seWords))	
				totFreq = reduce((lambda f1, f2: f1 + f2), frquencies)
				totFreq /= len(seWords)
				sentScores.append(totFreq)
 	
 			zippedSents = zip(range(len(sents)), sentScores)
 			sortedSents = sorted(zippedSents, key=takeSecond, reverse=True)
 			
 			#add to summary sentence list
 			tsIndex = sortedSents[0][0]
 			selSent = (tsIndex, sents[tsIndex], sortedSents[0][1])
 			topSentences.append(selSent) 
 			
 			#regularize frequencies
 			for w in sentWords[tsIndex]:
 				wordFreq[w] = wordFreq[w] * wordFreq[w]
 				
 			#remove from existing list
 			del sents[tsIndex]
 			del sentWords[tsIndex]
 		
 		topSentences = sorted(topSentences, key=takeFirst)	
 		return list(map(lambda ts: (ts[1], ts[2]), topSentences))

# tatent sematic analysis  summarizer		
class LatentSemSumm:
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["common.min.sentence.length"] = (5, None)
		defValues["Common.size"] = (5, None)
		defValues["common.byCount"] = (True, None)
		defValues["common.show.score"] = (False, None)
		defValues["lsi.num.topics"] = (5, None)
		defValues["lsi.chunk.size"] = (20000, None)
		defValues["lsi.decay"] = (1.0, None)
		defValues["lsi.distributed"] = (False, None)
		defValues["lsi.onepass"] = (True, None)
		defValues["lsi.power.iters"] = (2, None)
		defValues["lsi.extra.samples"] = (100, None)
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
 	
	# get config object
	def getConfig(self):
		return self.config

	def getSummary(self, filePath, text=None):
		minSentLength = self.config.getIntConfig("common.min.sentence.length")[0]
		docSent = DocSentences(filePath, minSentLength, self.verbose, text)
		sents = docSent.getSentences()

		#LSI model
		sentTokens = docSent.getSentencesAsTokens()
		dct = corpora.Dictionary(sentTokens)
		corpus = list(map(lambda st: dct.doc2bow(st), sentTokens))
		numTopics = self.config.getIntConfig("lsi.num.topics")[0]
		chunkSize = self.config.getIntConfig("lsi.chunk.size")[0]
		decay = self.config.getFloatConfig("lsi.decay")[0]
		distributed = self.config.getBooleanConfig("lsi.distributed")[0]
		onepass = self.config.getBooleanConfig("lsi.onepass")[0]
		powerIters = self.config.getIntConfig("lsi.power.iters")[0]
		extraSamples = self.config.getIntConfig("lsi.extra.samples")[0]
		lsi = models.LsiModel(corpus, id2word=dct, num_topics=numTopics, chunksize=chunkSize, decay=decay,\
			distributed=distributed, onepass=onepass, power_iters=powerIters, extra_samples=extraSamples)
		vecCorpus = lsi[corpus]

		#sort each vector by score
		sortedVecs = list(map(lambda i: list(), range(numTopics)))
		for i,dv in enumerate(vecCorpus):
			for sc in dv:
				isc = (i, abs(sc[1]))
				sortedVecs[sc[0]].append(isc)
		sortedVecs = list(map(lambda iscl: sorted(iscl,key=takeSecond,reverse=True), sortedVecs))	
		if self.verbose:
			print "num topics " + str(len(sortedVecs))
			print "lat vec length " + str(len(sortedVecs[0]))

		#select sentences
 		summSize  = self.config.getIntConfig("common.size")[0]
 		byCount = self.config.getBooleanConfig("common.byCount")[0]
 		if not byCount:
 			summSize = (len(sortedSents) * summSize) / 100
		numScans = (summSize / numTopics) + 1

		topSentences = []
		for i in range(numScans):
			for j in range(numTopics):
				vecs = sortedVecs[j]
				topSentences.append(vecs[i])
		topSentences = sorted(topSentences[:summSize], key=takeFirst)
		return list(map(lambda ts: (sents[ts[0]], ts[1]), topSentences))

# text rank summarizer		
class TextRankSumm:
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["Common.size"] = (5, None)
		defValues["common.byCount"] = (True, None)
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]

	# get config object
	def getConfig(self):
		return self.config

	def getSummary(self, filePath, text=None):
		if filePath:
			with open(filePath, 'r') as contentFile:
				content = contentFile.read()
		elif text:
			content = text
		else:
			raise ValueError("either file path or text must be provided")

 		summSize  = self.config.getIntConfig("common.size")[0]
 		byCount = self.config.getBooleanConfig("common.byCount")[0]
		if byCount:
			fraction = 0.2
			wordCount = summSize
		else:
			fraction = float(summSize) / 100
			wordCount = None
		return summarize(content, ratio=fraction, word_count=wordCount, split=True)

# sum basic summarizer		
class EmbeddingTextRankSumm:
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["common.min.sentence.length"] = (5, None)
		defValues["common.size"] = (5, None)
		defValues["common.byCount"] = (True, None)
		defValues["common.show.score"] = (False, None)
		defValues["etr.model.path"] = (None, "missing embedding model file path")
		defValues["etr.vec.size"] = (None, "missing embedding vector size")
		defValues["etr.max.missing.vec"] = (0.05, None)
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.totalWordCount = 0
		self.missingWordCount = 0
		
 	
	# get config object
	def getConfig(self):
		return self.config
			
	def getSummary(self, filePath, text=None):
		minSentLength = self.config.getIntConfig("common.min.sentence.length")[0]
		docSent = DocSentences(filePath, minSentLength, self.verbose, text)
		sents = docSent.getSentences()
		sentWords = docSent.getSentencesAsTokens()
		numSents = len(sents)
		vecSize = self.config.getIntConfig("etr.vec.size")[0]
		
		#glove embedding
		emModPath = self.config.getStringConfig("etr.model.path")[0]
		embeddings = {}
		with open(emModPath, 'r') as fi:
			for line in fi:
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings[word] = coefs
		
		#sentence vecs
		sentVecs = []
		for sw in sentWords:
			wv = list(map(lambda w: self.getWordVec(w, embeddings, vecSize), sw))
			sv = sum(wv) / len(sw)
			sentVecs.append(sv)
		if self.verbose:
			print "num of words  " + str(self.totalWordCount)
			print "num of missing word embedding " + str(self.missingWordCount)
		maxMissingVec = self.config.getFloatConfig("etr.max.missing.vec")[0]
		fracMissing = self.missingWordCount / float(self.totalWordCount)
		if fracMissing > maxMissingVec:
			print "too many missing vectore..... quitting"
			return []
			
		#similarity
		sentVec2d = np.array(sentVecs)
		simMat = cosine_similarity(sentVec2d)
		for i in range(numSents):
			simMat[i][i] = 0

		#page rank
		graph = nx.from_numpy_array(simMat)
		scores = nx.pagerank(graph)
		
		#top sentences
		summSize  = self.config.getIntConfig("common.size")[0]
 		byCount = self.config.getBooleanConfig("common.byCount")[0]
 		if not byCount:
 			summSize = (len(sortedSents) * summSize) / 100

		sentWithScores = enumerate(scores)
		sortedSents = sorted(sentWithScores, key=takeSecond, reverse=True)
		topSents = sortedSents[:summSize]
		topSents = sorted(topSents, key=takeFirst)
 		return list(map(lambda ts: (sents[ts[0]], ts[1]), topSents))

	def getWordVec(self, w, embeddings, vecSize):
		self.totalWordCount += 1
		if w in embeddings:
			v = embeddings.get(w)
		else:
			self.missingWordCount += 1
			v = np.zeros((vecSize,))
		return v	
		
# sentence selection by max marginal relevance
class MaxMarginalRelevance: 	
	def __init__(self, termFreq, byCount, normalized):
		self.termFreq = termFreq
		self.byCount = byCount
		self.normalized = normalized
	
	def select(self, sentsWithScore, numSel, regParam, noveltyAggr):
		#normalize scores
		mss = 0
		for sc in sentsWithScore:
			if sc[2] > msc:
				msc = sc[2]
		msc = float(msc)
		sentsWithScore = list(map(lambda sc: (sc[0], sc[1], sc[2]/msc), sentsWithScore))		


		selected = []
		for i in range(numSel):
			maxSc = 0
			nextSel = None
			nextVec = None
			for sc in sentsWithScore:
				words = sc[1]
				vec = termFreq.getVector(words, self.byCount, self.normalized)
				if len(selected) > 0:	
					dists = list(map(lambda se: cosineDistance(vec, se[4]), selected))
					if noveltyAggr == "max":
						novelty = max(dists)
					elif noveltyAggr == "min":
						novelty = min(dists)
					elif noveltyAggr == "avearge":
						novelty = sum(dists) / len(dists)
					else:
						raise ValueError("invalid novelty aggregator")
				else:
					novelty = 0
				newSc = regParam * sc[2] + (1.0 - regParam) * novelty
				if newSc > maxSc:
					maxSc = newSc
					nextSel = sc
					nextVec = vec
			next = (nextSel[0], nextSel[1], nextSel[2], maxSc, nextVec)
			selected.append(next)
			sentsWithScore.remove(nextSel)
				
		#sort by index
		selected = sorted(selected, key=takeFirst)	
		return selected

	
	
