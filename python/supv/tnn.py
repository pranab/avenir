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
import matplotlib.pyplot as plt
import numpy as np
import torch
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
		defValues["common.mode"] = ("training", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.preprocessing"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.data.out.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.num.hidden.units"] = (None, "missing number of hidden units")
		defValues["train.batch.size"] = (10, None)
		defValues["train.loss.reduction"] = ("sum", None)
		defValues["train.learning.rate"] = (.0001, None)
		defValues["train.num.iterations"] = (500, None) 
		
		super(ThreeLayerNetwork, self).__init__()
    	

	def buildModel(self):
		"""
    	Loads configuration and builds the various piecess necessary for the model
		"""
		self.numinp = len(self.config.getStringConfig("train.data.feature.fields")[0].split(","))
		self.numHidden = self.config.getIntConfig("train.num.hidden.units")[0]
		self.numOut = len(self.config.getStringConfig("train.data.out.fields")[0].split(","))
		self.batchSize = self.config.getIntConfig("train.batch.size")[0]
		self.lossRed = self.config.getStringConfig("train.loss.reduction")[0]
		self.learnRate = self.config.getFloatConfig("train.learnig.rate")[0]
		self.numIter = self.config.getIntConfig("train.num.iterations")[0]
    	
		self.linear1 = torch.nn.Linear(numinp, numHidden)
		self.linear2 = torch.nn.Linear(numHidden, numOut)
		self.lossFn = torch.nn.MSELoss(reduction=lossRed)
		(featData, outData) = self.prepTrainingData()
		self.featData = torch.from_numpy(featData)
		self.outData = torch.from_numpy(outData)
    

	def forward(self, x):
		"""
    	In the forward function we accept a Tensor of input data and we must return
    	a Tensor of output data. We can use Modules defined in the constructor as
    	well as arbitrary (differentiable) operations on Tensors.
		"""
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred

	def prepTrainingData(self):
		"""
		loads and prepares training data
		"""
		# parameters
		dataFile = self.config.getStringConfig("train.data.file")[0]
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
		return (featData, outData)

	@staticmethod
	def train(model):
		optimizer = torch.optim.SGD(model.parameters(), lr=model.learnRate)
		for t in range(model.numIter):
			# Forward pass: Compute predicted y by passing x to the model
			y_pred = model(self.featData)

			# Compute and print loss
			loss = model.lossFn(y_pred, model.outData)
			print(t, loss.item())

			# Zero gradients, perform a backward pass, and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()    	
    