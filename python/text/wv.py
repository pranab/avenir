#!/usr/bin/python
#!/usr/bin/python

import os
import sys
import nltk
from random import shuffle
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import jprops
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from util import *
from mlutil import *

# word2vec
class WordToVec:
	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["common.verbose"] = (False, None)
		defValues["common.model.directory"] = (None, "missing model dir")
		defValues["common.model.file"] = ("wv", None)
		defValues["common.model.full"] = (True, None)
		defValues["train.feature.size"] = (100, None)
		defValues["train.window.context"] = (20, None)
		defValues["train.min.word.count"] = (3, None)
		defValues["train.num.iter"] = (50, None)
		defValues["train.algo"] = (0, None)
		defValues["train.learn.rate"] = (0.025, None)
		defValues["train.hir.softmax"] = (0, None)
		defValues["train.neg.samp"] = (5, None)
		defValues["train.neg.samp.exp"] = (0.75, None)
		defValues["train.sample"] = (0.001, None)
		defValues["train.model.save"] = (False, None)
		
		self.config = Configuration(configFile, defValues)
		self.model = None
		self.wv = None
		
	# get config object
	def getConfig(self):
		return self.config
	
	#set config param
	def setConfigParam(self, name, value):
		self.config.setParam(name, value)

	# train model	
	def train(self, docs):
		featureSize = self.config.getIntConfig("train.feature.size")[0]
		windowContext = self.config.getIntConfig("train.window.context")[0]
		minWordCount = self.config.getIntConfig("train.min.word.count")[0]
		numIter = self.config.getIntConfig("train.num.iter")[0]
		sample = self.config.getFloatConfig("train.sample")[0]
		algo = self.config.getIntConfig("train.algo")[0]
		learnRate = self.config.getFloatConfig("train.learn.rate")[0]
		hSoftmax = self.config.getIntConfig("train.hir.softmax")[0]
		negSamp = self.config.getIntConfig("train.neg.samp")[0]
		negSampExp = self.config.getFloatConfig("train.neg.samp.exp")[0]

		self.model = Word2Vec(docs, size=featureSize, alpha=learnRate, window=windowContext, min_count=minWordCount, \
		sample=sample, iter=numIter, sg=algo, hs=hSoftmax, negative=negSamp)
		
		modelSave = self.config.getBooleanConfig("train.model.save")[0]
		if modelSave:
			self.model.save(self.getModelFilePath())
	
	# save vectors only	
	def saveVectors(self):
		path = self.getModelFilePath()
		if 	self.model is None:
			self.model = word2vec.load(path)
		self.model.wv.save(path)
		del self.model
	
	# find nearest words	
	def findSimilarWords(self, words, topn):
		self.initModel()		
		similarWords = {searchTerm: [item[0] for item in self.wv.most_similar([searchTerm], topn=topn)] \
                  for searchTerm in words}
		return similarWords
	
	# find nearest words	
	def findSimilarByWords(self, word, topn):
		self.initModel()		
		return self.wv.similar_by_word(word, topn=topn)
		
	# find nearest words	
	def findSimilarByVector(self, vector, topn):
		self.initModel()		
		return self.wv.similar_by_vector(vector, topn=topn)

	# similarity between 2 words
	def findSimilarity(word1, word2):
		self.initModel()
		return self.wv.similarity(word1, word2)
		
	# similarity between 2 list of words
	def findMultiWordSimilarity(words1, words2):
		self.initModel()
		return self.wv.n_similarity(words1, words2)

	# distance between 2 words
	def findDistance(word1, word2):
		self.initModel()
		return self.wv.distance(word1, word2)

	# word mover distance between 2 lists of words
	def findWmDistance(words1, words2):
		self.initModel()
		return 	self.wv.wmdistance(words1, words2)
		
	# initialize word vectors	
	def initModel(self):
		path = self.getModelFilePath()
		modelFull = self.config.getBooleanConfig("common.model.full")[0]
		if modelFull:
			if self.model is None:
				self.model = Word2Vec.load(path)
			self.wv = self.model.wv
		else:
			if self.wv is None:
				self.wv = KeyedVectors.load(path, mmap='r')
	
	# get model file path
	def getModelFilePath(self):
		modelDirectory = self.config.getStringConfig("common.model.directory")[0]
		modelFile = self.config.getStringConfig("common.model.file")[0]
		if modelFile is None:
			raise ValueError("missing model file name")
		modelFilePath = os.path.join(modelDirectory, modelFile)
		return modelFilePath
	
