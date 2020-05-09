#!/usr/local/bin/python3

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
import matplotlib
import random
import jprops
from sklearn.ensemble import GradientBoostingClassifier 
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from random import randint
from io import StringIO
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from pasearch import *
from bacl import *

# gradient boosting classification
class GradientBoostedTrees(object):
	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.preprocessing"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.data.class.field"] = (None, "missing class field ordinal")
		defValues["train.validation"] = ("kfold", None)
		defValues["train.num.folds"] = (5, None)
		defValues["train.min.samples.split"] = ("4", None)
		defValues["train.min.samples.leaf.gb"] = ("2", None)
		defValues["train.max.depth.gb"] = (3, None)
		defValues["train.max.leaf.nodes.gb"] = (None, None)
		defValues["train.max.features.gb"] = (None, None)
		defValues["train.learning.rate"] = (0.1, None)
		defValues["train.num.estimators.gb"] = (100, None)
		defValues["train.subsample"] = (1.0, None)
		defValues["train.loss"] = ("deviance", None)
		defValues["train.random.state"] = (None, None)
		defValues["train.verbose"] = (0, None)
		defValues["train.warm.start"] = (False, None)
		defValues["train.presort"] = ("auto", None)
		defValues["train.criterion"] = ("friedman_mse", None)
		defValues["train.success.criterion"] = ("error", None)
		defValues["train.model.save"] = (False, None)
		defValues["train.score.method"] = ("accuracy", None)
		defValues["train.search.param.strategy"] = (None, None)
		defValues["train.search.params"] = (None, None)
		defValues["predict.data.file"] = (None, None)
		defValues["predict.data.fields"] = (None, "missing data field ordinals")
		defValues["predict.data.feature.fields"] = (None, "missing data feature field ordinals")
		defValues["predict.use.saved.model"] = (False, None)
		defValues["validate.data.file"] = (None, "missing validation data file")
		defValues["validate.data.fields"] = (None, "missing validation data field ordinals")
		defValues["validate.data.feature.fields"] = (None, "missing validation data feature field ordinals")
		defValues["validate.data.class.field"] = (None, "missing class field ordinal")
		defValues["validate.use.saved.model"] = (False, None)
		defValues["validate.score.method"] = ("accuracy", None)

		self.config = Configuration(configFile, defValues)
		self.subSampleRate  = None
		self.featData = None
		self.clsData = None
		self.gbcClassifier = None
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		logFilePath = self.config.getStringConfig("common.logging.file")[0]
		logLevName = self.config.getStringConfig("common.logging.level")[0]
		self.logger = createLogger(__name__, logFilePath, logLevName)
		self.logger.info("********* starting session")
		
	# initialize config
	def initConfig(self, configFile, defValues):
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
		
	#get search parameter
	def getSearchParamStrategy(self):
		return self.config.getStringConfig("train.search.param.strategy")[0]

	def setModel(self, model):
		self.gbcClassifier = model
		
	# train model	
	def train(self):
		#build model
		self.buildModel()
		
		# training data
		if self.featData is None:
			(featData, clsData) = self.prepTrainingData()
			(self.featData, self.clsData) = (featData, clsData)
		else:
			(featData, clsData) = (self.featData, self.clsData)
		if self.subSampleRate is not None:
			(featData, clsData) = subSample(featData, clsData, self.subSampleRate, False)
			self.logger.info("subsample size  " + str(featData.shape[0]))
		
		# parameters
		modelSave = self.config.getBooleanConfig("train.model.save")[0]
		
		#train
		self.logger.info("...training model")
		self.gbcClassifier.fit(featData, clsData) 
		score = self.gbcClassifier.score(featData, clsData)  
		successCriterion = self.config.getStringConfig("train.success.criterion")[0]
		result = None
		if successCriterion == "accuracy":
			self.logger.info("accuracy with training data {:06.3f}".format(score))
			result = score
		elif successCriterion == "error":
			error = 1.0 - score
			self.logger.info("error with training data {:06.3f}".format(error))
			result = error
		else:
			raise ValueError("invalid success criterion")
			
		if modelSave:
			self.logger.info("...saving model")
			modelFilePath = self.getModelFilePath()
			joblib.dump(self.gbcClassifier, modelFilePath) 
		return result
		
	#train with k fold validation
	def trainValidate(self):
		#build model
		self.buildModel()

		# training data
		(featData, clsData) = self.prepTrainingData()
		
		#parameter
		validation = self.config.getStringConfig("train.validation")[0]
		numFolds = self.config.getIntConfig("train.num.folds")[0]
		successCriterion = self.config.getStringConfig("train.success.criterion")[0]
		scoreMethod = self.config.getStringConfig("train.score.method")[0]
		
		#train with validation
		self.logger.info("...training and kfold cross validating model")
		scores = cross_val_score(self.gbcClassifier, featData, clsData, cv=numFolds,scoring=scoreMethod)
		avScore = np.mean(scores)
		result = self.reportResult(avScore, successCriterion, scoreMethod)
		return result
		
	#train with k fold validation and search parameter space for optimum
	def trainValidateSearch(self):
		self.logger.info("...starting train validate with parameter search")
		searchStrategyName = self.getSearchParamStrategy()
		if searchStrategyName is not None:
			if searchStrategyName == "grid":
				searchStrategy = GuidedParameterSearch(self.verbose)
			elif searchStrategyName == "random":
				searchStrategy = RandomParameterSearch(self.verbose)
				maxIter = self.config.getIntConfig("train.search.max.iterations")[0]
				searchStrategy.setMaxIter(maxIter)
			elif searchStrategyName == "simuan":
				searchStrategy = SimulatedAnnealingParameterSearch(self.verbose)
				maxIter = self.config.getIntConfig("train.search.max.iterations")[0]
				searchStrategy.setMaxIter(maxIter)
				temp = self.config.getFloatConfig("train.search.sa.temp")[0]
				searchStrategy.setTemp(temp)
				tempRedRate = self.config.getFloatConfig("train.search.sa.temp.red.rate")[0]
				searchStrategy.setTempReductionRate(tempRedRate)
			else:
				raise ValueError("invalid paramtere search strategy")
		else:
			raise ValueError("missing search strategy")
				
		# add search params
		searchParams = self.config.getStringConfig("train.search.params")[0].split(",")
		searchParamNames = []
		extSearchParamNames = []
		if searchParams is not None:
			for searchParam in searchParams:
				paramItems = searchParam.split(":")
				extSearchParamNames.append(paramItems[0])
				
				#get rid name component search
				paramNameItems = paramItems[0].split(".")
				del paramNameItems[1]
				paramItems[0] = ".".join(paramNameItems)
				
				searchStrategy.addParam(paramItems)
				searchParamNames.append(paramItems[0])
		else:
			raise ValueError("missing search parameter list")
			
		# add search param data list for each param
		for (searchParamName,extSearchParamName)  in zip(searchParamNames,extSearchParamNames):
			searchParamData = self.config.getStringConfig(extSearchParamName)[0].split(",")
			searchStrategy.addParamVaues(searchParamName, searchParamData)
			
		# train and validate for various param value combination
		searchStrategy.prepare()
		paramValues = searchStrategy.nextParamValues()
		searchResults = []
		while paramValues is not None:
			self.logger.info("...next parameter set")
			paramStr = ""
			for paramValue in paramValues:
				self.setConfigParam(paramValue[0], str(paramValue[1]))
				paramStr = paramStr + paramValue[0] + "=" + str(paramValue[1]) + "  "
			result = self.trainValidate()
			searchStrategy.setCost(result)
			searchResults.append((paramStr, result))
			paramValues = searchStrategy.nextParamValues()
			
		# output
		self.logger.info("all parameter search results")
		for searchResult in searchResults:
			self.logger.info("{}\t{:06.3f}".format(searchResult[0], searchResult[1]))
		
		self.logger.info("best parameter search result")
		bestSolution = searchStrategy.getBestSolution()
		paramStr = ""
		for paramValue in bestSolution[0]:
			paramStr = paramStr + paramValue[0] + "=" + str(paramValue[1]) + "  "
		self.logger.info("{}\t{:06.3f}".format(paramStr, bestSolution[1]))
		return bestSolution
			
	#predict
	def validate(self):
		# create model
		useSavedModel = self.config.getBooleanConfig("validate.use.saved.model")[0]
		if useSavedModel:
			# load saved model
			self.logger.info("...loading model")
			modelFilePath = self.getModelFilePath()
			self.gbcClassifier = joblib.load(modelFilePath)
		else:
			# train model
			self.train()
		
		# prepare test data
		(featData, clsDataActual) = self.prepValidationData()
		
		#predict
		self.logger.info("...predicting")
		clsDataPred = self.gbcClassifier.predict(featData) 
		
		self.logger.info("...validating")
		#self.logger.info(clsData)
		scoreMethod = self.config.getStringConfig("validate.score.method")[0]
		if scoreMethod == "accuracy":
			accuracy = accuracy_score(clsDataActual, clsDataPred) 
			self.logger.info("accuracy:")
			self.logger.info(accuracy)
		elif scoreMethod == "confusionMatrix":
			confMatrx = confusion_matrix(clsDataActual, clsDataPred)
			self.logger.info("confusion matrix:")
			self.logger.info(confMatrx)

	 
	#predict
	def predictx(self):
		# create model
		useSavedModel = self.config.getBooleanConfig("predict.use.saved.model")[0]
		if useSavedModel:
			# load saved model
			self.logger.info("...loading model")
			modelFilePath = self.getModelFilePath()
			self.gbcClassifier = joblib.load(modelFilePath)
		else:
			# train model
			self.train()
		
		# prepare test data
		featData = self.prepPredictData()
		
		#predict
		self.logger.info("...predicting")
		clsData = self.gbcClassifier.predict(featData) 
		self.logger.info(clsData)

	#predict with in memory data
	def predict(self, recs=None):
		# create model
		self.prepModel()
		
		#input record
		#input record
		if recs:
			#passed record
			featData = self.prepStringPredictData(recs)
			if (featData.ndim == 1):
				featData = featData.reshape(1, -1)
		else:
			#file
			featData = self.prepPredictData()
		
		#predict
		self.logger.info("...predicting")
		clsData = self.gbcClassifier.predict(featData) 
		return clsData

	#predict probability with in memory data
	def predictProb(self, recs):
		# create model
		self.prepModel()
		
		#input record
		if type(recs) is str:
			featData = self.prepStringPredictData(recs)
		else:
			featData = recs
		#self.logger.info(featData.shape)
		if (featData.ndim == 1):
			featData = featData.reshape(1, -1)
		
		#predict
		self.logger.info("...predicting class probability")
		clsData = self.gbcClassifier.predict_proba(featData) 
		return clsData
	
	#preparing model
	def prepModel(self):
		useSavedModel = self.config.getBooleanConfig("predict.use.saved.model")[0]
		if (useSavedModel and not self.gbcClassifier):
			# load saved model
			self.logger.info("...loading saved model")
			modelFilePath = self.getModelFilePath()
			self.gbcClassifier = joblib.load(modelFilePath)
		else:
			# train model
			self.train()
		return self.gbcClassifier
		
	#prepare string predict data
	def prepStringPredictData(self, recs):
		frecs = StringIO(recs)
		featData = np.loadtxt(frecs, delimiter=',')
		#self.logger.info(featData)
		return featData
	
	#loads and prepares training data
	def prepTrainingData(self):
		# parameters
		dataFile = self.config.getStringConfig("train.data.file")[0]
		fieldIndices = self.config.getStringConfig("train.data.fields")[0]
		if not fieldIndices is None:
			fieldIndices = strToIntArray(fieldIndices, ",")
		featFieldIndices = self.config.getStringConfig("train.data.feature.fields")[0]
		if not featFieldIndices is None:
			featFieldIndices = strToIntArray(featFieldIndices, ",")
		classFieldIndex = self.config.getIntConfig("train.data.class.field")[0]

		#training data
		(data, featData) = loadDataFile(dataFile, ",", fieldIndices, featFieldIndices)
		clsData = extrColumns(data, classFieldIndex)
		clsData = np.array([int(a) for a in clsData])
		return (featData, clsData)

	#loads and prepares training data
	def prepValidationData(self):
		# parameters
		dataFile = self.config.getStringConfig("validate.data.file")[0]
		fieldIndices = self.config.getStringConfig("validate.data.fields")[0]
		if not fieldIndices is None:
			fieldIndices = strToIntArray(fieldIndices, ",")
		featFieldIndices = self.config.getStringConfig("validate.data.feature.fields")[0]
		if not featFieldIndices is None:
			featFieldIndices = strToIntArray(featFieldIndices, ",")
		classFieldIndex = self.config.getIntConfig("validate.data.class.field")[0]

		#training data
		(data, featData) = loadDataFile(dataFile, ",", fieldIndices, featFieldIndices)
		clsData = extrColumns(data, classFieldIndex)
		clsData = [int(a) for a in clsData]
		return (featData, clsData)

	#loads and prepares training data
	def prepPredictData(self):
		# parameters
		dataFile = self.config.getStringConfig("predict.data.file")[0]
		if dataFile is None:
			raise ValueError("missing prediction data file")
		fieldIndices = self.config.getStringConfig("predict.data.fields")[0]
		if not fieldIndices is None:
			fieldIndices = strToIntArray(fieldIndices, ",")
		featFieldIndices = self.config.getStringConfig("predict.data.feature.fields")[0]
		if not featFieldIndices is None:
			featFieldIndices = strToIntArray(featFieldIndices, ",")

		#training data
		(data, featData) = loadDataFile(dataFile, ",", fieldIndices, featFieldIndices)
		
		return featData
	
	# get model file path
	def getModelFilePath(self):
		modelDirectory = self.config.getStringConfig("common.model.directory")[0]
		modelFile = self.config.getStringConfig("common.model.file")[0]
		if modelFile is None:
			raise ValueError("missing model file name")
		modelFilePath = modelDirectory + "/" + modelFile
		return modelFilePath
	
	# report result
	def reportResult(self, score, successCriterion, scoreMethod):
		if successCriterion == "accuracy":
			self.logger.info("average " + scoreMethod + " with k fold cross validation {:06.3f}".format(score))
			result = score
		elif successCriterion == "error":
			error = 1.0 - score
			self.logger.info("average error with k fold cross validation {:06.3f}".format(error))
			result = error
		else:
			raise ValueError("invalid success criterion")
		return result
	
	# builds model object
	def buildModel(self):
		self.logger.info("...building gradient boosted tree model")
		# parameters
		minSamplesSplit = self.config.getStringConfig("train.min.samples.split")[0]
		minSamplesSplit = typedValue(minSamplesSplit)
		minSamplesLeaf = self.config.getStringConfig("train.min.samples.leaf.gb")[0]
		minSamplesLeaf = typedValue(minSamplesLeaf)
		#minWeightFractionLeaf = self.config.getFloatConfig("train.min.weight.fraction.leaf.gb")[0]
		(maxDepth, maxLeafNodes) = self.config.eitherOrIntConfig("train.max.depth.gb", "train.max.leaf.nodes.gb")
		maxFeatures = self.config.getStringConfig("train.max.features.gb")[0]
		maxFeatures = typedValue(maxFeatures)
		learningRate = self.config.getFloatConfig("train.learning.rate")[0]
		numEstimators = self.config.getIntConfig("train.num.estimators.gb")[0]
		subsampleFraction = self.config.getFloatConfig("train.subsample")[0]	
		lossFun = self.config.getStringConfig("train.loss")[0]
		randomState = self.config.getIntConfig("train.random.state")[0]
		verboseOutput = self.config.getIntConfig("train.verbose")[0]
		warmStart = self.config.getBooleanConfig("train.warm.start")[0]
		presort = self.config.getStringConfig("train.presort")
		if (presort[1]):
			presortChoice = presort[0]
		else:
			presortChoice = presort[0].lower() == "true"
		splitCriterion = self.config.getStringConfig("train.criterion")[0]	
	
		#classifier
		self.gbcClassifier = GradientBoostingClassifier(loss=lossFun, learning_rate=learningRate, n_estimators=numEstimators, 
		subsample=subsampleFraction, min_samples_split=minSamplesSplit, 
		min_samples_leaf=minSamplesLeaf, min_weight_fraction_leaf=0.0, max_depth=maxDepth,  
		init=None, random_state=randomState, max_features=maxFeatures, verbose=verboseOutput, 
		max_leaf_nodes=maxLeafNodes, warm_start=warmStart, presort=presortChoice)


	
	
	
