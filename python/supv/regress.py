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
from io import StringIO
from sklearn.model_selection import cross_val_score
import joblib
from random import randint
from io import StringIO
from sklearn.linear_model import LinearRegression
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from pasearch import *

class BaseRegressor(object):
	"""
	base regression class
	"""
	
	def __init__(self, configFile, defValues):
		"""
		intializer
		"""
		defValues["common.mode"] = ("train", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.scale.file.path"] = (None, "missing scale file path")
		defValues["common.preprocessing"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.data.out.field"] = (None, "missing out field ordinal")	

		self.config = Configuration(configFile, defValues)
		self.featData = None
		self.outData = None
		self.regressor = None
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.mode = self.config.getBooleanConfig("common.mode")[0]
		logFilePath = self.config.getStringConfig("common.logging.file")[0]
		logLevName = self.config.getStringConfig("common.logging.level")[0]
		self.logger = createLogger(__name__, logFilePath, logLevName)
		self.logger.info("********* starting session")
	
	def initConfig(self, configFile, defValues):
		"""
		initialize config
		"""
		self.config = Configuration(configFile, defValues)
	
	def getConfig(self):
		"""
		get config object
		"""
		return self.config
	
	def setConfigParam(self, name, value):
		"""
		set config param
		"""
		self.config.setParam(name, value)

	def getMode(self):
		"""
		get mode
		"""
		return self.mode

	def train(self):
		"""
		train model
		"""
		#build model
		self.buildModel()
		
		# training data
		if self.featData is None:
			(featData, outData) = self.prepData("train")
			(self.featData, self.outData) = (featData, outData)
		else:
			(featData, outData) = (self.featData, self.outData)
		
		# parameters
		modelSave = self.config.getBooleanConfig("train.model.save")[0]
		
		#train
		self.logger.info("...training model")
		self.regressor.fit(featData, outData) 
		rsqScore = self.regressor.score(featData, outData)
		coef = self.regressor.coef_
		intc = self.regressor.intercept_
		result = (rsqScore, intc, coef)
		
		if modelSave:
			self.logger.info("...saving model")
			modelFilePath = self.getModelFilePath()
			joblib.dump(self.regressor, modelFilePath) 
		return result
		
	def validate(self):
		# create model
		self.prepModel()	
			
		# prepare test data
		(featData, outDataActual) = self.prepData("validate")
		
		#predict
		self.logger.info("...predicting")
		outDataPred = self.regressor.predict(featData) 
		
		#error
		rsqScore = self.regressor.score(featData, outDataActual)
		result = (outDataPred, rsqScore)
		return result

	def predict(self):
		"""
		predict using trained model
		"""
		# create model
		self.prepModel()
		
		# prepare test data
		featData = self.prepData("predict")[0]
		
		#predict
		self.logger.info("...predicting")
		outData = self.regressor.predict(featData) 
		return outData

	def prepData(self, mode):
		"""
		loads and prepares data for training and validation
		"""
		# parameters
		key = mode  + ".data.file"
		dataFile = self.config.getStringConfig(key)[0]
		
		key = mode  + ".data.fields"
		fieldIndices = self.config.getStringConfig(key)[0]
		if not fieldIndices is None:
			fieldIndices = strToIntArray(fieldIndices, ",")
			
		
		key = mode  + ".data.feature.fields"
		featFieldIndices = self.config.getStringConfig(key)[0]
		if not featFieldIndices is None:
			featFieldIndices = strToIntArray(featFieldIndices, ",")
		
		if not mode == "predict":	
			key = mode  + ".data.out.field"
			outFieldIndex = self.config.getIntConfig(key)[0]

		#load data
		(data, featData) = loadDataFile(dataFile, ",", fieldIndices, featFieldIndices)
		if (self.config.getStringConfig("common.preprocessing")[0] == "scale"):
			featData = sk.preprocessing.scale(featData)
		outData = None
		if not mode == "predict":
			outData = extrColumns(data, outFieldIndex)
		return (featData, outData)
		
	def prepModel(self):
		"""
		load saved model or train model
		"""
		useSavedModel = self.config.getBooleanConfig("predict.use.saved.model")[0]
		if (useSavedModel and not self.regressor):
			# load saved model
			self.logger.info("...loading saved model")
			modelFilePath = self.getModelFilePath()
			self.regressor = joblib.load(modelFilePath)
		else:
			# train model
			self.train()
	
class LinearRegressor(BaseRegressor):
	"""
	linear regression
	"""
	def __init__(self, configFile):
		defValues = {}
		defValues["train.normalize"] = (False, None)	

		super(LinearRegressor, self).__init__(configFile, defValues)

	def buildModel(self):
		"""
		builds model object
		"""
		self.logger.info("...building linear regression model")
		normalize = self.config.getBooleanConfig("train.normalize")[0]
		self.regressor = LinearRegression(normalize=normalize)

class ElasticNetRegressor(BaseRegressor):
	"""
	elastic net regression
	"""
	def __init__(self, configFile):
		defValues = {}
		defValues["train.alpha"] = (1.0, None)	
		defValues["train.loneratio"] = (0.5, None)
		defValues["train.normalize"] = (False, None)	
		defValues["train.precompute"] = (False, None)	
		defValues["train.max.iter"] = (1000, None)	
		defValues["train.tol"] = (0.0001, None)	
		defValues["train.random.state"] = (None, None)	
		defValues["train.selection"] = ("cyclic", None)	

		super(ElasticNetRegressor, self).__init__(configFile, defValues)

	def buildModel(self):
		"""
		builds model object
		"""
		self.logger.info("...building elastic net regression model")
		alpha = self.config.getFloatConfig("train.alpha")[0]
		loneratio = self.config.getFloatConfig("train.loneratio")[0]
		normalize = self.config.getBooleanConfig("train.normalize")[0]
		precompute = self.config.getBooleanConfig("train.precompute")[0]
		maxIter = self.config.getIntConfig("train.max.iter")[0]
		tol = self.config.getFloatConfig("train.tol")[0]
		randState = self.config.getIntConfig("train.random.state")[0]
		selection = self.config.getIntConfig("train.selection")[0]
		
		self.regressor =  ElasticNet(alpha=alpha, l1_ratio=loneratio, normalize=normalize, precompute=precompute, 
		max_iter=maxIter, tol=tol, random_state=randState, selection=selection)
		
		
