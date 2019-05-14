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
from random import shuffle
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import pickle
import jprops
from scipy import spatial
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from util import *
from mlutil import *

# word2vec
class DocToVec:
	def __init__(self, configFile):
		#train.window 5
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["common.verbose"] = (False, None)
		defValues["common.model.directory"] = (None, "missing model dir")
		defValues["common.model.file"] = ("wv", None)
		defValues["common.model.full"] = (True, None)
		defValues["common.model.full"] = (True, None)
		defValues["train.data.dir"] = (None, "missing training data directory")
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.text.granularity"] = ("document", None)
		defValues["train.model.save"] = (False, None)
		defValues["train.min.sentence.length"] = (10, None)
		defValues["train.distr.model"] = (1, None)
		defValues["train.vector.size"] = (50, None)
		defValues["train.window"] = (5, None)
		defValues["train.alpha"] = (0.025, None)
		defValues["train.alpha.min"] = (0.0001, None)
		defValues["train.seed"] = (1, None)
		defValues["train.vocab.max"] = (None, None)
		defValues["train.sample"] = (None, None)
		defValues["train.workers"] = (3, None)
		defValues["train.epochs"] = (5, None)
		defValues["train.batch.words"] = (10000, None)
		defValues["train.hierarch.softmax"] = (0, None)
		defValues["train.negative"] = (5, None)
		defValues["train.neg.samp.exp"] = (0.75, None)
		defValues["train.distr.model.mean"] = (1, None)
		defValues["train.distr.model.concat"] = (0, None)
		defValues["train.distr.model.tag.count"] = (1, None)
		defValues["train.dbow.words"] = (0, None)
		defValues["train.trim.rule"] = (None, None)
		defValues["generate.data.dir"] = (None, "missing generation data directory")
		defValues["generate.data.file"] = (None, "missing generation data file")
		defValues["generate.vector.dir"] = (None, "missing generated vector directory")
		defValues["generate.vector.file"] = (None, "missing generated vector file")
		defValues["generate.vector.save"] = (False, None)


		self.config = Configuration(configFile, defValues)
		self.model = None

	# get config object
	def getConfig(self):
		return self.config
	
	# set config param
	def setConfigParam(self, name, value):
		self.config.setParam(name, value)

	# train model	
	def train(self, docs):
		numWorkers = self.config.getIntConfig("train.workers")[0]
		vectorSize = self.config.getIntConfig("train.vector.size")[0]
		numEpochs = self.config.getIntConfig("train.epochs")[0]
		alpha = self.config.getFloatConfig("train.alpha")[0]
		windowSize = self.config.getIntConfig("train.window")[0]
		seed = self.config.getIntConfig("train.seed")[0]
		hirSoftMax = self.config.getIntConfig("train.hierarch.softmax")[0]
		negative = self.config.getIntConfig("train.negative")[0]
		negSampExp = self.config.getFloatConfig("train.neg.samp.exp")[0]
		minAlpha = self.config.getFloatConfig("train.alpha.min")[0]
		distrModel = self.config.getIntConfig("train.distr.model")[0]
		distrModelMean = self.config.getIntConfig("train.distr.model.mean")[0]
		dbowWords = self.config.getIntConfig("train.dbow.words")[0]
		distrModelConcat = self.config.getIntConfig("train.distr.model.concat")[0]
		distrModelTagCount = self.config.getIntConfig("train.distr.model.tag.count")[0]
		self.model = Doc2Vec(documents=docs,workers=numWorkers, vector_size=vectorSize, epochs=numEpochs, alpha=alpha,\
			window=windowSize, seed=seed, hs=hirSoftMax, negative=negative,ns_exponent=negSampExp,\
			min_alpha=minAlpha, dm_mean=distrModelMean, dm=distrModel, dbow_words=dbowWords,\
			dm_concat=distrModelConcat,dm_tag_count=distrModelTagCount)

		modelSave = self.config.getBooleanConfig("train.model.save")[0]
		if modelSave:
			self.model.save(self.getModelFilePath())

	# get vector embedding
	def getDocEmbedding(self, doc):
		self.initModel()
		return self.model.infer_vector(doc)


	# get vector embedding multiple docs
	def getDocEmbeddings(self, docs):
		vectors = [self.getDocEmbedding(doc) for doc in docs]
		return vectors

	# get model file path
	def getModelFilePath(self):
		modelDirectory = self.config.getStringConfig("common.model.directory")[0]
		modelFile = self.config.getStringConfig("common.model.file")[0]
		if modelFile is None:
			raise ValueError("missing model file name")
		modelFilePath = os.path.join(modelDirectory, modelFile)
		return modelFilePath

	# initialize word vectors	
	def initModel(self):		
		if self.model is None:
			path = self.getModelFilePath()
			self.model = Doc2Vec.load(path)

# manage vector embedding
class VectorEmbedding:
	def __init__(self, path, granularity):
		self.path = path
		self.granularity = granularity
		self.vectors = None
	
	#generate vectors
	def getVectors(self, tokens, doc2Vec):
		self.vectors = doc2Vec.getDocEmbeddings(tokens)
		for v in self.vectors:
			print v

	# cosine distances
	def getCosineDistances(self, indx):
		this = self.vectors[indx]
		#distances = list(map(lambda other: spatial.distance.cosine(this, other), self.vectors))	
		distances = list(map(lambda other: 1.0 - cosineSimilarity(this, other), self.vectors))	
		distances = [(i,v) for i,v in enumerate(distances)]
		distances.sort(key=takeSecond)
		return distances

	# save 
	def save(self, saveDir, saveFile):
		saveFilePath = os.path.join(saveDir, saveFile)
		with open(saveFilePath, "w") as sf:
			pickle.dump(self, sf)

	# load 
	@staticmethod
	def load(saveDir, saveFile):
		saveFilePath = os.path.join(saveDir, saveFile)
		with open(saveFilePath, "r") as sf:
			vecEmbed = pickle.load(sf)
		return vecEmbed
		

