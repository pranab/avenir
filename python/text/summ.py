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
from sklearn.decomposition import NMF
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from util import *
from mlutil import *
from preprocess import *
nltk.download('punkt')
	
#base classifier class
class BaseSummarizer(object):
	def __init__(self, configFile, defValues):
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
	
	#initialize config	
	def initConfig(self, configFile, defValues):
		self.config = Configuration(configFile, defValues)
	
	# get config object
	def getConfig(self):
		return self.config

	# top sentence count
	def getTopCount(self, sents):
		# top count
 		summSize  = self.config.getIntConfig("common.size")[0]
 		byCount = self.config.getBooleanConfig("common.byCount")[0]
 		if not byCount:
 			summSize = (len(sents) * summSize) / 100
 		print "summSize " + str(summSize)
		return summSize
	
	# summarize everything
	def summarizeAll(self, sents):
		return list(map(lambda s: (s, 1.0), sents))	

# text summarizer based on term frequency
class TermFreqSumm(BaseSummarizer):
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["common.min.sentence.length"] = (5, None)
		defValues["common.size"] = (5, None)
		defValues["common.byCount"] = (True, None)
		defValues["common.show.score"] = (False, None)
		defValues["tf.length.normalizer"] = ("linear", None)
		defValues["tf.diversify"] = (False, None)
		defValues["tf.diversify.byCount"] = (True, None)
		defValues["tf.diversify.normalized"] = (True, None)
		defValues["tf.diversify.regularizer"] = (0.7, None)
		defValues["tf.diversify.aggr"] = ("average", None)
		super(TermFreqSumm, self).__init__(configFile, defValues)

	# get summary sentences		
	def getSummary(self, filePath, text=None):
		minSentLength = self.config.getIntConfig("common.min.sentence.length")[0]
		normalizer = self.config.getStringConfig("tf.length.normalizer")[0]

		docSent = DocSentences(filePath, minSentLength, self.verbose, text)
		sents = docSent.getSentences()
		summSize = self.getTopCount(sents)

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
 		
		diversify = self.config.getBooleanConfig("tf.diversify")[0]
		if diversify:
			#sentence index, words and score
			termTable.getWordFreq()
			zippedSentsWithWords = zip(zippedSents, sentWords)
			sentsWithScore = list(map(lambda zsw: (zsw[0][0], zsw[1], zsw[0][1]), zippedSentsWithWords))
			
			#diversify
			byCount = self.config.getBooleanConfig("tf.diversify.byCount")[0]
			normalized = self.config.getBooleanConfig("tf.diversify.normalized")[0]
			regParam = self.config.getFloatConfig("tf.diversify.regularizer")[0]
			diversityAggr = self.config.getStringConfig("tf.diversify.aggr")[0]
			mmr = MaxMarginalRelevance(termTable, byCount, normalized)
			topSents = mmr.select(sentsWithScore, summSize, regParam, diversityAggr)
			
			#include sentence, original score, diversify base score
			topSents =  list(map(lambda sc: (sents[sc[0]], sc[2],  sc[3]), topSents))	
			return topSents
		else:
 			# sort by decreasing score	
 			sortedSents = sorted(zippedSents, key=takeSecond, reverse=True)
 			print "after soerting num sentences " + str(len(sortedSents))

 			topSents = sortedSents[:summSize]
 		
 			#sort sentence by position 
 			topSents = sorted(topSents, key=takeFirst)	
 			return list(map(lambda ts: (sents[ts[0]], ts[1]), topSents))
 		
# sum basic summarizer		
class SumBasicSumm(BaseSummarizer):
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["common.min.sentence.length"] = (5, None)
		defValues["Common.size"] = (5, None)
		defValues["common.byCount"] = (True, None)
		defValues["common.show.score"] = (False, None)
		super(SumBasicSumm, self).__init__(configFile, defValues)
 			
	def getSummary(self, filePath, text=None):
		minSentLength = self.config.getIntConfig("common.min.sentence.length")[0]
		docSent = DocSentences(filePath, minSentLength, self.verbose, text)
		sents = docSent.getSentences()
		summSize = self.getTopCount(sents)
		
		# term count table for all words
		termTable = TfIdf(None, False)
		sentWords = docSent.getSentencesAsTokens()
		for seWords in sentWords:
			termTable.countDocWords(seWords)
		wordFreq = termTable.getWordFreq()
 		
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

# latent sematic analysis  summarizer		
class LatentSemSumm(BaseSummarizer):
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
		super(LatentSemSumm, self).__init__(configFile, defValues)
 	
	def getSummary(self, filePath, text=None):
		minSentLength = self.config.getIntConfig("common.min.sentence.length")[0]
		docSent = DocSentences(filePath, minSentLength, self.verbose, text)
		sents = docSent.getSentences()
		summSize = self.getTopCount(sents)

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
		numScans = (summSize / numTopics) + 1

		topSentences = []
		for i in range(numScans):
			for j in range(numTopics):
				vecs = sortedVecs[j]
				topSentences.append(vecs[i])
		topSentences = sorted(topSentences[:summSize], key=takeFirst)
		return list(map(lambda ts: (sents[ts[0]], ts[1]), topSentences))

# non negative matrix factorization summarizer		
class NonNegMatFactSumm(BaseSummarizer):
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["common.min.sentence.length"] = (5, None)
		defValues["Common.size"] = (5, None)
		defValues["common.byCount"] = (True, None)
		defValues["common.show.score"] = (False, None)
		defValues["nmf.num.topics"] = (5, None)
		defValues["nmf.init"] = (None, None)
		defValues["nmf.solver"] = ("cd", None)
		defValues["nmf.beta.loss"] = ("frobenius", None)
		defValues["nmf.tol"] = (0.0001, None)
		defValues["nmf.max.iter"] = (200, None)
		defValues["nmf.random.state"] = (None, None)
		defValues["nmf.alpha"] = (0.0, None)
		defValues["nmf.l1.ratio"] = (0.0, None)
		defValues["nmf.shuffle"] = (False, None)
		super(NonNegMatFactSumm, self).__init__(configFile, defValues) 	

	def getSummary(self, filePath, text=None):
		minSentLength = self.config.getIntConfig("common.min.sentence.length")[0]
		docSent = DocSentences(filePath, minSentLength, self.verbose, text)
		sents = docSent.getSentences()
		summSize = self.getTopCount(sents)
		numTopics = self.config.getIntConfig("nmf.num.topics")[0]
		
		
		# term count table for all words
		termTable = TfIdf(None, False)
		sentTokens = docSent.getSentencesAsTokens()
		for seToks in sentTokens:
			termTable.countDocWords(seToks)
			
		#vectorize
		sentVecs = list(map(lambda seToks: termTable.getVector(seToks, True, False), sentTokens))
		
		# NMF
		init = self.config.getStringConfig("nmf.init")[0]
		solver = self.config.getStringConfig("nmf.solver")[0]
		betaLoss = self.config.getStringConfig("nmf.beta.loss")[0]
		betaLoss = typedValue(betaLoss)
		tol = self.config.getFloatConfig("nmf.tol")[0]
		maxIter = self.config.getIntConfig("nmf.max.iter")[0]
		randomState = self.config.getIntConfig("nmf.random.state")[0]
		alpha = self.config.getFloatConfig("nmf.alpha")[0]
		l1Ratio = self.config.getFloatConfig("nmf.l1.ratio")[0]
		shuffle = self.config.getBooleanConfig("nmf.shuffle")[0]
		mat = np.array(sentVecs)
		model = NMF(n_components=numTopics, init=init, solver=solver, beta_loss=betaLoss, tol=tol,\
			max_iter=maxIter, random_state=randomState, alpha=alpha, l1_ratio=l1Ratio, shuffle=shuffle)
		vecCorpus = model.fit_transform(mat)
		H = model.components_
		if self.verbose:
			print vecCorpus.shape
			print vecCorpus
		
		#sort each vector by score
		sortedVecs = []
		for i in range(numTopics):
			col = list(enumerate(list(vecCorpus[:,i])))
			col = sorted(col,key=takeSecond,reverse=True)
			sortedVecs.append(col)
		
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
class TextRankSumm(BaseSummarizer):
	def __init__(self, configFile):
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.data.directory"] = (None, "missing data dir")
		defValues["common.data.file"] = (None, "missing data file")
		defValues["Common.size"] = (5, None)
		defValues["common.byCount"] = (True, None)
		super(TextRankSumm, self).__init__(configFile, defValues) 	

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
class EmbeddingTextRankSumm(BaseSummarizer):
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
		self.totalWordCount = 0
		self.missingWordCount = 0
		super(EmbeddingTextRankSumm, self).__init__(configFile, defValues) 	
 	
	def getSummary(self, filePath, text=None):
		minSentLength = self.config.getIntConfig("common.min.sentence.length")[0]
		docSent = DocSentences(filePath, minSentLength, self.verbose, text)
		sents = docSent.getSentences()
		summSize = self.getTopCount(sents)
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
			if self.verbose:
				print "missing vector for " + w
		return v	
		
# sentence selection by max marginal relevance
class MaxMarginalRelevance: 	
	def __init__(self, termFreq, byCount, normalized):
		self.termFreq = termFreq
		self.byCount = byCount
		self.normalized = normalized
	
	def select(self, sentsWithScore, numSel, regParam, diversityAggr):
		#normalize scores
		msc = 0
		for sc in sentsWithScore:
			if sc[2] > msc:
				msc = sc[2]
		msc = float(msc)
		sentsWithScore = list(map(lambda sc: (sc[0], sc[1], sc[2]/msc), sentsWithScore))		

		#diversify
		selected = []
		for i in range(numSel):
			maxSc = 0
			nextSel = None
			nextVec = None
			for sc in sentsWithScore:
				words = sc[1]
				vec = self.termFreq.getVector(words, self.byCount, self.normalized)
				if len(selected) > 0:	
					dists = list(map(lambda se: cosineDistance(vec, se[4]), selected))
					if diversityAggr == "max":
						diversity = max(dists)
					elif diversityAggr == "min":
						diversity = min(dists)
					elif diversityAggr == "average":
						diversity = sum(dists) / len(dists)
					else:
						raise ValueError("invalid diversity aggregator " + diversityAggr)
				else:
					diversity = 0

				#diversity based score
				newSc = regParam * sc[2] + (1.0 - regParam) * diversity
				if newSc > maxSc:
					maxSc = newSc
					nextSel = sc
					nextVec = vec
			
			#add new score and vector 
			next = (nextSel[0], nextSel[1], nextSel[2], maxSc, nextVec)
			selected.append(next)
			sentsWithScore.remove(nextSel)
				
		#sort by index
		selected = sorted(selected, key=takeFirst)	
		return selected

	
	
