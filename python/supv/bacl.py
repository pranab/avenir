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
import matplotlib
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random
import jprops
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from pasearch import *

class BaseClassifier(object):
	config = None
	subSampleRate  = None
	featData = None
	clsData = None
	classifier = None

	def __init__(self, configFile, defValues):
		self.config = Configuration(configFile, defValues)
	
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
			print "subsample size  " + str(featData.shape[0])
		
		# parameters
		modelSave = self.config.getBooleanConfig("train.model.save")[0]
		
		#train
		print "...training model"
		self.classifier.fit(featData, clsData) 
		score = self.classifier.score(featData, clsData)  
		successCriterion = self.config.getStringConfig("train.success.criterion")[0]
		result = None
		if successCriterion == "accuracy":
			print "accuracy with training data %.3f" %(score)
			result = score
		elif successCriterion == "error":
			error = 1.0 - score
			print "error with training data %.3f" %(error)
			result = error
		else:
			raise ValueError("invalid success criterion")
			
		if modelSave:
			print "...saving model"
			modelFilePath = self.getModelFilePath()
			joblib.dump(self.classifier, modelFilePath) 
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
		print "...training and kfold cross validating model"
		scores = cross_val_score(self.classifier, featData, clsData, cv=numFolds,scoring=scoreMethod)
		avScore = np.mean(scores)
		result = self.reportResult(avScore, successCriterion, scoreMethod)
		return result

	#train with k fold validation and search parameter space for optimum
	def trainValidateSearch(self):
		print "...starting train validate with parameter search"
		searchStrategyName = self.getSearchParamStrategy()
		if searchStrategyName is not None:
			if searchStrategyName == "guided":
				searchStrategy = GuidedParameterSearch()
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
				paramNameItems = paramItems[0].split(".")
				del paramNameItems[1]
				paramItems[0] = ".".join(paramNameItems)
				searchStrategy.addParam(paramItems)
				searchParamNames.append(paramItems[0])
		else:
			raise ValueError("missing search parameter list")
			
		# add search param data
		for (searchParamName,extSearchParamName)  in zip(searchParamNames,extSearchParamNames):
			searchParamData = self.config.getStringConfig(extSearchParamName)[0].split(",")
			searchStrategy.addParamVaues(searchParamName, searchParamData)
			
		# train and validate for various param value combination
		searchStrategy.prepare()
		paramValues = searchStrategy.nextParamValues()
		searchResults = []
		while paramValues is not None:
			print "...next parameter set"
			paramStr = ""
			for paramValue in paramValues:
				self.setConfigParam(paramValue[0], str(paramValue[1]))
				paramStr = paramStr + paramValue[0] + "=" + str(paramValue[1]) + "  "
			result = self.trainValidate()
			searchStrategy.setCost(result)
			searchResults.append((paramStr, result))
			paramValues = searchStrategy.nextParamValues()
			
		# output
		print "all parameter search results"
		for searchResult in searchResults:
			print "%s\t%.3f" %(searchResult[0], searchResult[1])
		
		print "best parameter search result"
		bestSolution = searchStrategy.getBestSolution()
		paramStr = ""
		for paramValue in bestSolution[0]:
			paramStr = paramStr + paramValue[0] + "=" + str(paramValue[1]) + "  "
		print "%s\t%.3f" %(paramStr, bestSolution[1])
		return bestSolution

	#predict
	def validate(self):
		# create model
		useSavedModel = self.config.getBooleanConfig("validate.use.saved.model")[0]
		if useSavedModel:
			# load saved model
			print "...loading model"
			modelFilePath = self.getModelFilePath()
			self.classifier = joblib.load(modelFilePath)
		else:
			# train model
			self.train()
		
		# prepare test data
		(featData, clsDataActual) = self.prepValidationData()
		
		#predict
		print "...predicting"
		clsDataPred = self.classifier.predict(featData) 
		
		print "...validating"
		#print clsData
		scoreMethod = self.config.getStringConfig("validate.score.method")[0]
		if scoreMethod == "accuracy":
			accuracy = accuracy_score(clsDataActual, clsDataPred) 
			print "accuracy:"
			print accuracy
		elif scoreMethod == "confusionMatrix":
			confMatrx = confusion_matrix(clsDataActual, clsDataPred)
			print "confusion matrix:"
			print confMatrx

	#predict
	def predict(self):
		# create model
		useSavedModel = self.config.getBooleanConfig("predict.use.saved.model")[0]
		if useSavedModel:
			# load saved model
			print "...loading model"
			modelFilePath = self.getModelFilePath()
			self.classifier = joblib.load(modelFilePath)
		else:
			# train model
			self.train()
		
		# prepare test data
		featData = self.prepPredictData()
		
		#predict
		print "...predicting"
		print "classes " + str(self.classifier.classes_)
		predictProb = self.config.getBooleanConfig("train.predict.probability")[0]
		if predictProb:
			print "...predicting class probability"
			clsprData = self.classifier.predict_proba(featData)
			print clsprData
		else:
			print "...predicting class labels"
			clsData = self.classifier.predict(featData) 
			print clsData


	#auto train	
	def autoTrain(self):
		maxTestErr = self.config.getFloatConfig("train.auto.max.test.error")[0]
		maxErr = self.config.getFloatConfig("train.auto.max.error")[0]
		maxErrDiff = self.config.getFloatConfig("train.auto.max.error.diff")[0]
		
		self.config.setParam("train.model.save", "False")
		
		#train, validate and serach optimum parameter
		result = self.trainValidateSearch()
		testError = result[1]
			
		#subsample training size to match train size for k fold validation
		numFolds = self.config.getIntConfig("train.num.folds")[0]
		self.subSampleRate = float(numFolds - 1) / numFolds

		#train only with optimum parameter values
		for paramValue in result[0]:
			pName = paramValue[0]
			pValue = paramValue[1]
			print pName + "  " + pValue
			self.setConfigParam(pName, pValue)
		trainError = self.train()
		
		if testError < maxTestErr:
			# criteria based on test error only
			print "Successfullt trained. Low test error level"
			status = 1
		else:
			# criteria based on bias error and generalization error
			avError = (trainError + testError) / 2
			diffError = testError - trainError
			print "Auto training  completed: training error %.3f test error: %.3f" %(trainError, testError)
			print "Average of test and training error: %.3f test and training error diff: %.3f" %(avError, diffError)  
			if diffError > maxErrDiff:
				# high generalization error
				if avError > maxErr:
					# high bias error
					print "High generalization error and high error. Need larger training data set and increased model complexity"
					status = 4
				else:
					# low bias error
					print "High generalization error. Need larger training data set"
					status = 3
			else:
				# low generalization error
				if avError > maxErr:
					# high bias error
					print "Converged, but with high error rate. Need to increase model complexity"
					status = 2
				else:
					# low bias error
					print "Successfullt trained. Low generalization error and low bias error level"
					status = 1
				
		if status == 1:
			#train final model, use all data and save model
			print "...training the final model"
			self.config.setParam("train.model.save", "True")
			self.subSampleRate  = None
			trainError = self.train()
			print "training error in final model %.3f" %(trainError)
		
		return status
			
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
		featData = self.scaleOrNormalize(featData, False)
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
		featData = self.scaleOrNormalize(featData, True)
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
		featData = self.scaleOrNormalize(featData, True)
		return featData

	# scale or normalize data
	def scaleOrNormalize(self, featData, useSavedScale):
		preprocess = self.config.getStringConfig("common.preprocessing")[0]
		if preprocess == "scale":
			scaleFilePath = self.config.getStringConfig("common.scale.file.path")[0]
			if useSavedScale:
				scaler = joblib.load(scaleFilePath)
				featData = scaler.transform(featData)
			else:
				scaler = sk.preprocessing.StandardScaler().fit(featData)
				featData = scaler.transform(featData)
				joblib.dump(scaler, scaleFilePath) 
		elif preprocess == "normalize":
			featData = sk.preprocessing.normalize(featData, norm='l2')
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
			print "average " + scoreMethod + " with k fold cross validation %.3f" %(score)
			result = score
		elif successCriterion == "error":
			error = 1.0 - score
			print "average error with k fold cross validation %.3f" %(error)
			result = error
		else:
			raise ValueError("invalid success criterion")
		return result
		
		
