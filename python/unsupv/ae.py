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
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import sklearn as sk
import matplotlib
import random
import jprops
from random import randint
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *

class AutoEncoder(nn.Module):
	def __init__(self):
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
		defValues["train.num.hidden.units"] = (None, "missing number of hidden units for each layer")
		defValues["train.learning.rate"] = (.0001, None)
		defValues["train.weight.decay"] = (.00001, None)
		defValues["train.betas"] = ("0.9, 0.999", None)
		defValues["train.eps"] = (1e-8, None)
		defValues["train.momentum"] = (0, None)
		defValues["train.dampening"] = (0, None)
		defValues["train.batch.size"] = (128, None)
		defValues["train.momentum.nesterov"] = (False, None)
		defValues["train.num.iterations"] = (500, None) 
		defValues["train.optimizer"] = ("adam", None) 

		super(AutoEncoder, self).__init__()
        
	def buildModel(self):
		"""
    	Loads configuration and builds the various piecess necessary for the model
		"""
		self.numinp = len(self.config.getStringConfig("train.data.feature.fields")[0].split(","))
		self.numHidden = self.config.getStringConfig("train.num.hidden.units")[0].split(",")
		self.numHidden = strToIntArray(self.numHidden, ",")
		numLayers = len(self.numHidden)
		self.learnRate = self.config.getFloatConfig("train.learnig.rate")[0]
		self.batchSize = self.config.getIntConfig("train.batch.size")[0]
		self.weightDecay = self.config.getFloatConfig("train.weight.decay")[0]
		self.betas = self.config.getStringConfig("train.betas")[0].split(",")
		self.betas = strToFloatArray(self.betas, ",")
		self.eps = self.config.getFloatConfig("train.eps")[0]
		self.momentum = self.config.getFloatConfig("train.momentum")[0]
		self.dampening = self.config.getFloatConfig("train.dampening")[0]
		self.momentumNesterov = self.config.getBooleanConfig("train.momentum.nesterov")[0]
		self.numIter = self.config.getIntConfig("train.num.iterations")[0]
		self.optimizer = self.config.getStringConfig("train.optimizer")[0]
		
		featData = self.prepTrainingData()
		self.featData = torch.from_numpy(featData)
		self.dataloader = DataLoader(self.featData, batch_size=self.batchSize, shuffle=True)
		
		#encoder
		inSize = self.numinp
		outSize = self.numHidden[0]
		enModules = list()
		for i in range(numLayers-1):
			enModules.append(nn.Linear(inSize, outSize))
			enModules.append(nn.ReLU(True))
			inSize = outSize
			outSize = self.numHidden[i+1]
		enModules.append(nn.Linear(inSize, outSize))
		self.encoder = nn.ModuleList(enModules)
			
        #decoder
		deModules = list()
		for i in reversed(range(1, numLayers)):
			inSize = self.numHidden[i]
			outSize = self.numHidden[i-1]
			deModules.append(nn.Linear(inSize, outSize))
			deModules.append(nn.ReLU(True))
		deModules.append(nn.Linear(self.numHidden[0], self.numinp))
		self.decoder = nn.ModuleList(deModules)

	def forward(self, x):
		"""
		forward pass
		"""
		x = self.encoder(x)
		x = self.decoder(x)
		return x

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
		featData = loadFeatDataFile(dataFile, ",", fieldIndices)
		if (self.config.getStringConfig("common.preprocessing")[0] == "scale"):
			featData = sk.preprocessing.scale(featData)
		return featData
		
	@staticmethod
	def train(model):
		"""
		train model
		"""
		# optimizer
		if self.optimizer == "adam":
			betas = (self.betas[0], self.betas[1]) 
			optimizer = torch.optim.Adam(model.parameters(), lr=model.learnRate, betas=betas, eps = self.eps,\
				weight_decay=model.weightDecay)
		elif self.optimizer == "sgd":
			optimizer = torch.optim.SGD(model.parameters(), momentum=self.momentum, dampening=self.dampening,\
				weight_decay=model.weightDecay, nesterov=self.momentumNesterov)
		else:
			raise ValueError("invalid optimizer type")
			
		criterion = nn.MSELoss()
		for it in range(model.numIter):
			for data in self.dataloader:
				output = model(data)
				loss = criterion(output, data)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			print('epoch [{}/{}], loss {:.4f}'.format(it + 1, model.numIter, loss.data[0]))

