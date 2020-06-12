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
import torch
from torch.autograd import Variable
import sklearn as sk
import matplotlib
import random
import jprops
from sklearn.ensemble import RandomForestClassifier 
from random import randint
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *

class ThreeLayerNetwork(torch.nn.Module):
	def __init__(self, configFile):
		"""
    	In the constructor we instantiate two nn.Linear modules and assign them as
    	member variables.
		"""
		defValues = dict()
		defValues["common.mode"] = ("training", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.preprocessing"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.data.out.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.num.hidden.units.one"] = (None, "missing number of hidden units")
		defValues["train.activation.one"] = ("relu", None) 
		defValues["train.num.hidden.units.two"] = (None, None)
		defValues["train.activation.two"] = (None, None) 
		defValues["train.batch.size"] = (10, None)
		defValues["train.loss.reduction"] = ("mean", None)
		defValues["train.learning.rate"] = (.0001, None)
		defValues["train.num.iterations"] = (500, None) 
		defValues["train.optimizer"] = ("sgd", None) 
		defValues["train.lossFn"] = ("mse", None) 
		defValues["valid.data.file"] = (None, None)
		defValues["valid.accuracy.metric"] = (None, None)
		self.config = Configuration(configFile, defValues)
		
		super(ThreeLayerNetwork, self).__init__()
    	

	def buildModel(self):
		"""
    	Loads configuration and builds the various piecess necessary for the model
		"""
		self.verbose = self.config.getStringConfig("common.verbose")[0]
		numinp = len(self.config.getStringConfig("train.data.feature.fields")[0].split(","))
		numHiddenOne = self.config.getIntConfig("train.num.hidden.units.one")[0]
		activationOne = self.config.getStringConfig("train.activation.one")[0]
		numHiddenTwo = self.config.getIntConfig("train.num.hidden.units.two")[0]
		activationTwo = self.config.getStringConfig("train.activation.two")[0]
		numOut = len(self.config.getStringConfig("train.data.out.fields")[0].split(","))
		self.batchSize = self.config.getIntConfig("train.batch.size")[0]
		lossRed = self.config.getStringConfig("train.loss.reduction")[0]
		learnRate = self.config.getFloatConfig("train.learning.rate")[0]
		self.numIter = self.config.getIntConfig("train.num.iterations")[0]
		optimizer = self.config.getStringConfig("train.optimizer")[0]
		lossFn = self.config.getStringConfig("train.lossFn")[0]
		self.accMetric = self.config.getStringConfig("valid.accuracy.metric")[0]
   	
		self.linear1 = torch.nn.Linear(numinp, numHiddenOne)
		self.act1 = self.createActivation(activationOne)
		if numHiddenTwo:
			#two hidden layers
			self.linear2 = torch.nn.Linear(numHiddenOne, numHiddenTwo)
			self.act2 = self.createActivation(activationTwo)
			self.linear3 = torch.nn.Linear(numHiddenTwo, numOut)
			print("2 hidden layers")
		else:
			#one hidden layer
			self.linear2 = None
			self.act2 = None
			self.linear3 = torch.nn.Linear(numHiddenOne, numOut)
			print("1 hidden layer")

		#training data
		dataFile = self.config.getStringConfig("train.data.file")[0]
		(featData, outData) = self.prepData(dataFile)
		self.featData = Variable(torch.from_numpy(featData))
		self.outData = Variable(torch.from_numpy(outData))

		#validation data
		dataFile = self.config.getStringConfig("valid.data.file")[0]
		(featDataV, outDataV) = self.prepData(dataFile)
		self.validFeatData = Variable(torch.from_numpy(featDataV))
		self.validOutData = outDataV

		#loss function
		if lossFn == "mse":
			self.lossFn = torch.nn.MSELoss(reduction=lossRed)
		elif lossFn == "ce":
			self.lossFn = torch.nn.CrossEntropyLoss(reduction=lossRed)
		elif lossFn == "lone":
			self.lossFn = torch.nn.L1Loss(reduction=lossRed)
		else:
			exitWithMsg("invalid loss function")
    
    	#optimizer
		if optimizer == "sgd":
			self.optimizer = torch.optim.SGD(self.parameters(), lr=learnRate)
		elif optimizer == "adam":
			self.optimizer = torch.optim.Adam(self.parameters(), lr=learnRate)
		elif optimizer == "rmsprop":
			self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learnRate)
		else:
			exitWithMsg("invalid optimizer")

	def createActivation(self, activation):
		"""
		create activation
		"""
		if activation is None:
			act = None
		elif activation == "relu":
			act = torch.nn.ReLU()
		elif activation == "tanh":
			act = torch.nn.Tanh()
		elif activation == "sigmoid":
			act = torch.nn.Sigmoid()
		else:
			exitWithMsg("invalid activation function")
		return act

	def forward(self, x):
		"""
    	In the forward function we accept a Tensor of input data and we must return
    	a Tensor of output data. We can use Modules defined in the constructor as
    	well as arbitrary (differentiable) operations on Tensors.
		"""
		y = self.linear1(x)
		y = self.act1(y)
		if  self.linear2 is not None:
			y = self.linear2(y)
			y = self.act2(y)
		y = self.linear3(y)
		return y

	def prepData(self, dataFile):
		"""
		loads and prepares  data
		"""
		# parameters
		fieldIndices = self.config.getStringConfig("train.data.fields")[0]
		fieldIndices = strToIntArray(fieldIndices, ",")
		featFieldIndices = self.config.getStringConfig("train.data.feature.fields")[0]
		featFieldIndices = strToIntArray(featFieldIndices, ",")

		#training data
		(data, featData) = loadDataFile(dataFile, ",", fieldIndices, featFieldIndices)
		if (self.config.getStringConfig("common.preprocessing")[0] == "scale"):
			featData = sk.preprocessing.scale(featData)
		outFieldIndices = self.config.getStringConfig("train.data.out.fields")[0]
		outFieldIndices = strToIntArray(outFieldIndices, ",")
		outData = data[:,outFieldIndices]	
		return (featData.astype(np.float32), outData.astype(np.float32))

	@staticmethod
	def trainModel(model):

		# train mode
		model.train()
		for t in range(model.numIter):

	
			# Forward pass: Compute predicted y by passing x to the model
			yPred = model(model.featData)

			# Compute and print loss
			loss = model.lossFn(yPred, model.outData)
			if t % 20 == 0:
				print(t, loss.item())

			# Zero gradients, perform a backward pass, and update the weights.
			model.optimizer.zero_grad()
			loss.backward()
			model.optimizer.step()    	

		#validate
		model.eval()
		yPred = model(model.validFeatData)
		yPred = yPred.data.cpu().numpy()
		yActual = model.validOutData
		if model.verbose:
			result = np.concatenate((yPred, yActual), axis = 1)
			print("predicted  actual")
			print(result)
		
		score = perfMetric(model.accMetric, yActual, yPred)
		print(formatFloat(3, score, "perf score"))



    