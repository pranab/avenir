#!/Users/pranab/Tools/anaconda/bin/python

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
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.linear_model
import matplotlib
import random
import jprops
from sklearn.neural_network import BernoulliRBM
from sklearn.externals import joblib
from random import randint
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *

# restricted boltzman machine
class RestrictedBoltzmanMachine:

	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("train", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["train.data.file"] = (None, None)
		defValues["train.data.fields"] = (None, None)
		defValues["train.model.save"] = (False, None)
		defValues["train.num.components"] = (None, "missing num of hidden components")
		defValues["train.learning.rate"] = (0.1, None)
		defValues["train.batch.size"] = (10, None)
		defValues["train.num.iter"] = (10, None)
		defValues["train.random.state"] = (None, None)
		defValues["train.model.save"] = (False, None)
		defValues["analyze.data.file"] = (None, None)
		defValues["analyze.data.fields"] = (None, None)
		defValues["analyze.use.saved.model"] = (True, None)
		defValues["analyze.missing.initial.count"] = (10, None)
		defValues["analyze.missing.iter.count"] = (100, None)
		defValues["analyze.missing.validate"] = (False, None)
		defValues["analyze.missing.validate.file.path"] = (None, None)
		
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.analyzeData = None
		self.model = None
	
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
	def train(self):
		#build model
		self.buildModel()
		
		dataFile = self.config.getStringConfig("train.data.file")[0]
		fieldIndices = self.config.getStringConfig("train.data.fields")[0]
		if not fieldIndices is None:
			fieldIndices = strToIntArray(fieldIndices, ",")
		data = np.loadtxt(dataFile, delimiter=",", usecols=fieldIndices)
		self.model.fit(data)

		modelSave = self.config.getBooleanConfig("train.model.save")[0]
		if modelSave:
			self.saveModel()

	# reconstruct
	def reconstruct(self):
		self.getModel()
		self.getAnalyzeData()
		recon =  self.model.gibbs(self.analyzeData)
		recNum = list()
		for r in recon:
			recNum.append(list(map(lambda b: 1 if b else 0, r)))
		return recNum
		
	# analyze data
	def getAnalyzeData(self):
		if self.analyzeData is None:
			dataFile = self.config.getStringConfig("analyze.data.file")[0]
			fieldIndices = self.config.getStringConfig("analyze.data.fields")[0]
			if not fieldIndices is None:
				fieldIndices = strToIntArray(fieldIndices, ",")
			self.analyzeData = np.loadtxt(dataFile, delimiter=",", usecols=fieldIndices)
		return self.analyzeData
	
	# get model file path
	def getModelFilePath(self):
		modelDirectory = self.config.getStringConfig("common.model.directory")[0]
		modelFile = self.config.getStringConfig("common.model.file")[0]
		if modelFile is None:
			raise ValueError("missing model file name")
		modelFilePath = modelDirectory + "/" + modelFile
		return modelFilePath
	
	# save model
	def saveModel(self):
		modelSave = self.config.getBooleanConfig("train.model.save")[0]
		if modelSave:
			print "...saving model"
			modelFilePath = self.getModelFilePath()
			joblib.dump(self.model, modelFilePath) 
	
	# gets model
	def getModel(self):
		useSavedModel = self.config.getBooleanConfig("analyze.use.saved.model")[0]
		if self.model is None:
			if useSavedModel:
				# load saved model
				print "...loading model"
				modelFilePath = self.getModelFilePath()
				self.model = joblib.load(modelFilePath)
			else:
				# train model
				self.train()
	
		
	# builds model object
	def buildModel(self):
		numComp = self.config.getIntConfig("train.num.components")[0]
		learnRate = self.config.getFloatConfig("train.learning.rate")[0]
		batchSize = self.config.getIntConfig("train.batch.size")[0]
		numIter = self.config.getIntConfig("train.num.iter")[0]
		randomState = self.config.getIntConfig("train.random.state")[0]
		self.model = BernoulliRBM(n_components=numComp, learning_rate=learnRate, batch_size=batchSize,\
			n_iter=numIter, verbose=self.verbose, random_state=randomState)
		
		
		