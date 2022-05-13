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
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import sklearn as sk
import matplotlib
import random
import jprops
from random import randint
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from tnn import *


class FeedForwardTwinNetwork(FeedForwardNetwork):
	"""
	siamese twin feef forward network
	"""
	def __init__(self, configFile):
		defValues = dict()
		defValues["train.twin.data.feat.count"] = (None, "missing training data feature count")
		defValues["train.twin.final.layer.size"] = (None, "missing final layer size")
		defValues["train.twin.crossenc"] = (False, None)
		super(FeedForwardTwinNetwork, self).__init__(configFile, defValues)

	def buildModel(self):
		"""
    	Loads configuration and builds the various piecess necessary for the model
		"""
		super().buildModel()
		
		#final fully connected after merge
		finSize = self.config.getIntConfig("train.twin.final.layer.size")[0]
		self.fcOut = torch.nn.Linear(finSize, 1)
		self.crossEnc =  self.config.getBooleanConfig("train.twin.crossenc")[0]

	def forward(self, x1, x2):
		"""
    	Go through layers twice
		"""
		y1 = self.layers(x1)	
		y2 = self.layers(x2)
		if self.crossEnc:
			y = torch.cat((y1, y2), 1)
		else:
			y = torch.abs(y1 - y2)	
		y = self.fcOut(y)
		return y
		
	@staticmethod
	def batchTrain(model):
		"""
		train with batch data
		"""
		feCount = model.config.getIntConfig("train.data.feat.count")[0]
		feOne = model.featData[:,:feCount]
		feTwo = model.featData[:,feCount:]
		trainData = TensorDataset(feOne, feTwo, model.outData)
		trainDataLoader = DataLoader(dataset=trainData, batch_size=model.batchSize, shuffle=True)
		epochIntv = model.config.getIntConfig("train.epoch.intv")[0]

		# train mode
		model.train()
 
		if model.trackErr:
			trErr = list()
			vaErr = list()
		#epoch
		for t in range(model.numIter):
			#batch
			b = 0
			epochLoss = 0.0
			for xOneBatch, xTwoBatch, yBatch in trainDataLoader:
	
				# Forward pass: Compute predicted y by passing x to the model
				yPred = model(xOneBatch, xTwoBatch)
				
				# Compute and print loss
				loss = model.lossFn(yPred, yBatch)
				if model.verbose and t % epochIntv == 0 and b % model.batchIntv == 0:
					print("epoch {}  batch {}  loss {:.6f}".format(t, b, loss.item()))
				
				if model.trackErr and model.batchIntv == 0:
					epochLoss += loss.item()
				
				#error tracking at batch level
				if model.trackErr and model.batchIntv > 0 and b % model.batchIntv == 0:
					trErr.append(loss.item())
					vloss = FeedForwardNetwork.evaluateModel(model)
					vaErr.append(vloss)

				# Zero gradients, perform a backward pass, and update the weights.
				model.optimizer.zero_grad()
				loss.backward()
				model.optimizer.step()    	
				b += 1
			
			#error tracking at epoch level
			if model.trackErr and model.batchIntv == 0:
				epochLoss /= len(trainDataLoader)
				trErr.append(epochLoss)
				vloss = FeedForwardNetwork.evaluateModel(model)
				vaErr.append(vloss)
			
		#validate
		model.eval()
		vaFeOne = model.validFeatData[:,:feCount]
		vaFeTwo = model.validFeatData[:,feCount:]
		yPred = model(vaFeOne, vaFeTwo)
		yPred = yPred.data.cpu().numpy()
		yActual = model.validOutData
		if model.verbose:
			vsize = yPred.shape[0]
			print("\npredicted \t\t actual")
			for i in range(vsize):
				print(str(yPred[i]) + "\t" + str(yActual[i]))
		
		score = perfMetric(model.accMetric, yActual, yPred)
		print(yActual)
		print(yPred)
		print(formatFloat(3, score, "perf score"))

		#save
		modelSave = model.config.getBooleanConfig("train.model.save")[0]
		if modelSave:
			FeedForwardNetwork.saveCheckpt(model)

		if model.trackErr:
			FeedForwardNetwork.errorPlot(model, trErr, vaErr)
			
		return score
		
		

		
