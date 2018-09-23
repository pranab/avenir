#!/usr/bin/python

import os
import sys
import nltk
from nltk.corpus import movie_reviews
from random import shuffle
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
import jprops
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from util import *
from mlutil import *

# gradient boosting classification
class LatentDirichletAllocation:
	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["common.verbose"] = (False, None)
		defValues["common.model.directory"] = (None, "missing model dir")
		defValues["common.model.file"] = ("lda", None)
		defValues["common.dictionary.file"] = ("dict", None)
		defValues["train.data.dir"] = (None, "missing training data dir")
		defValues["train.num.topics"] = (None, "missing num of topics")
		defValues["train.num.pass"] = (100, None)
		defValues["train.min.doc.occurence.abs"] = (5, None)
		defValues["train.max.doc.occurence.fraction"] = (0.5, None)
		defValues["train.keep.most.frequent.count"] = (100000, None)
		defValues["train.remove.most.frequent.count"] = (2, None)
		defValues["train.model.save"] = (False, None)
		defValues["train.plot.perplexity"] = (False, None)
		defValues["analyze.data.dir"] = (None, "missing analyze data dir")
		defValues["analyze.doc.topic.odds.ratio"] = (1.5, None)
		defValues["analyze.topic.word.odds.ratio"] = (1.5, None)
		defValues["analyze.topic.word.top.max"] = (20, None)
		defValues["analyze.base.word.distr.file"] = (None, None)
		defValues["analyze.word.cross.entropy.filter"] = (False, None)

		self.config = Configuration(configFile, defValues)
		self.ldaModel = None
		self.dictionary = None
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.docTopics = []
		self.singleNumOfTopics = True

	# get config object
	def getConfig(self):
		return self.config
	
	#set config param
	def setConfigParam(self, name, value):
		self.config.setParam(name, value)
	
	#get mode
	def getMode(self):
		return self.config.getStringConfig("common.mode")[0]

	# train model	
	def train(self, docs):
		# create dictionary
		self.dictionary = self.createDictionary(docs)

		# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
		docTermMatrix = [self.dictionary.doc2bow(doc) for doc in docs]

		#build model
		topicRange = self.config.getIntListConfig("train.num.topics", ",")[0]
		numPass = self.config.getIntConfig("train.num.pass")[0]
		topicMin = topicRange[0]
		if len(topicRange) == 2:
			topicMax = topicRange[1]
		else:
			topicMax = topicRange[0]
		topics = []
		perplex = []
		self.singleNumOfTopics = (topicMin == topicMax)
		for numTopics in range(topicMin, topicMax+1):
			print "**num of topics " + str(numTopics)

			# Running and Trainign LDA model on the document term matrix.
			self.buildModel(docTermMatrix, numTopics,  numPass)
			
			# output
			if self.verbose:
				print "--topics"
				print(self.ldaModel.print_topics(num_topics=numTopics, num_words=5))

			self.docTopics = []
			if self.verbose:
				print "--doc topic vector"
			for doc in docTermMatrix:
				docLda = self.ldaModel[doc]
				self.docTopics.append(docLda)
				if self.verbose:
					print docLda

			# perplexity
			perplexity = self.ldaModel.log_perplexity(docTermMatrix)
			if self.verbose:
				print "--perplexity %.6f" %(perplexity)

			topics.append(numTopics)
			perplex.append(perplexity)

		modelSave = self.config.getBooleanConfig("train.model.save")[0]
		if modelSave:
			self.dictionary.save(self.getModelFilePath("common.dictionary.file"))
			self.ldaModel.save(self.getModelFilePath("common.model.file"))

		return (topics, perplex)

	# get topic distr for docs
	def getDocTopics(self):
		return 	self.docTopics
	
	# only one number of topics
	def isSingleNumOfTopics(self):
		return self.singleNumOfTopics

	# train model	
	def analyze(self, docs):
		# load dictionary and model
		self.dictionary = Dictionary.load(self.getModelFilePath("common.dictionary.file"))
		self.ldaModel = LdaModel.load(self.getModelFilePath("common.model.file"))

		# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
		docTermMatrix = [self.dictionary.doc2bow(doc) for doc in docs]

		docTopicDistr = self.getDocumentTopics(docTermMatrix)
		return docTopicDistr

	# train model incrementally	
	def update(self, docs):
		# load dictionary and model
		self.dictionary = Dictionary.load(self.getModelFilePath("common.dictionary.file"))
		self.ldaModel = LdaModel.load(self.getModelFilePath("common.model.file"))

		# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
		docTermMatrix = [self.dictionary.doc2bow(doc) for doc in docs]

		numPass = self.config.getIntConfig("train.num.pass")[0]
		self.ldaModel.update(docTermMatrix, passes=numPasses)

		docTopicDistr = self.getDocumentTopics(docTermMatrix)
		return docTopicDistr

	# create dictionary
	def createDictionary(self, docs):
		# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
		dictionary = corpora.Dictionary(docs)
		#print dictionary

		# filter out some terms
		kwargs = {}
		minDoc = self.config.getIntConfig("train.min.doc.occurence.abs")[0]
		if minDoc is not None:
			kwargs["no_below"] = minDoc
		maxDoc = self.config.getFloatConfig("train.max.doc.occurence.fraction")[0]
		if maxDoc is not None:
			kwargs["no_above"] = maxDoc
		keep = self.config.getIntConfig("train.keep.most.frequent.count")[0]
		if keep is not None:
			kwargs["keep_n"] = keep
		dictionary.filter_extremes(**kwargs)

		# remove most frequent terms
		remove = self.config.getIntConfig("train.remove.most.frequent.count")[0]
		if remove is not None:
			dictionary.filter_n_most_frequent(remove)
		print dictionary
		return dictionary

	# build model
	def buildModel(self, docTermMatrix, numTopics,  numPasses):
		"""  LDA for topic modeling  """
		self.ldaModel = gensim.models.ldamodel.LdaModel(docTermMatrix, num_topics=numTopics, id2word=self.dictionary, passes=numPasses)
		return self.ldaModel

	# doc topic distr
	def getDocumentTopics(self, docTermMatrix):
		""" return topic distribution for doc """
		return self.ldaModel.get_document_topics(docTermMatrix)

	# topic word distr
	def getTopicTerms(self, topicId, topN):
		""" return word distribution for a topic """
		idto = self.dictionary.id2token
		tiDistr =  self.ldaModel.get_topic_terms(topicId, topN)
		toDistr = [(idto[ti[0]], ti[1])  for ti in tiDistr]
		return toDistr

	# all topic word distr
	def getAllTopicTerms(self):
		""" return all topic / word distribution """
		return self.ldaModel.get_topics()

	# get model file path
	def getModelFilePath(self, fileNameParam):
		modelDirectory = self.config.getStringConfig("common.model.directory")[0]
		modelFile = self.config.getStringConfig(fileNameParam)[0]
		if modelFile is None:
			raise ValueError("missing model file name")
		modelFilePath = os.path.join(modelDirectory, modelFile)
		return modelFilePath


