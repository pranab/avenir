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
from util import *
from mlutil import *

# gradient boosting classification
class LatentDirichletAllocation:
	config = None
	ldaModel = None
	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["common.model.directory"] = (None, "missing model dir")
		defValues["common.model.file"] = ("lda", None)
		defValues["train.data.dir"] = (None, "missing training data dir")
		defValues["train.num.topics"] = (None, "missing num of topics")
		defValues["train.num.pass"] = (100, None)
		defValues["train.model.save"] = (False, None)
		defValues["train.plot.perplexity"] = (False, None)
		defValues["analyze.data.dir"] = (None, "missing analyze data dir")

		self.config = Configuration(configFile, defValues)

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
	def train(self, docTermMatrix, dictionary):
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
		for numTopics in range(topicMin, topicMax+1):
			print "**num of topics " + str(numTopics)

			# Running and Trainign LDA model on the document term matrix.
			self.buildModel(docTermMatrix, numTopics, dictionary, numPass)
			
			# output
			print "--topics"
			print(self.ldaModel.print_topics(num_topics=numTopics, num_words=5))

			print "--doc latent vector"
			for doc in docTermMatrix:
				docLda = self.ldaModel[doc]
				print docLda

			# perplexity
			perplexity = self.ldaModel.log_perplexity(docTermMatrix)
			print "--perplexity %.6f" %(perplexity)

			topics.append(numTopics)
			perplex.append(perplexity)

		modelSave = self.config.getBooleanConfig("train.model.save")[0]
		if modelSave:
			self.ldaModel.save(self.getModelFilePath())

		return (topics, perplex)

	# build model
	def buildModel(self, docTermMatrix, numTopics, dictionary, numPasses):
		"""  LDA for topic modeling  """
		self.ldaModel = gensim.models.ldamodel.LdaModel(docTermMatrix, num_topics=numTopics, id2word=dictionary, passes=numPasses)
		return self.ldaModel

	# doc topic distr
	def getDocumentTopics(self, docTermMatrix):
		""" return topic distribution for doc """
		return self.ldaModel.get_document_topics(docTermMatrix)

	# topic word distr
	def getTopicTerms(self, topicId, topN):
		""" return word distribution for a topic """
		return self.ldaModel.get_topic_terms(topicId, topN)

	# all topic word distr
	def getAllTopicTerms(self):
		""" return all topic / word distribution """
		return self.ldaModel.get_topics()

	# get model file path
	def getModelFilePath(self):
		modelDirectory = self.config.getStringConfig("common.model.directory")[0]
		modelFile = self.config.getStringConfig("common.model.file")[0]
		if modelFile is None:
			raise ValueError("missing model file name")
		modelFilePath = os.path.join(modelDirectory, modelFile)
		return modelFilePath


