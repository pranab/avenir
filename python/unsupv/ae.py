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
from sklearn import metrics
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../supv"))
from util import *
from mlutil import *
from tnn import FeedForwardNetwork
from stats import *

class AutoEncoder(nn.Module):
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
		defValues["common.scaling.method"] = ("zscale", None)
		defValues["common.scaling.minrows"] = (50, None)
		defValues["common.verbose"] = (False, None)
		defValues["common.device"] = ("cpu", None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.out.fields"] = (None, "missing training data output field ordinals")
		defValues["train.num.input"] = (None, "missing number of input")
		defValues["train.num.hidden.units"] = (None, "missing number of hidden units for each layer")
		defValues["train.encoder.activations"] = (None, "missing encoder activation")
		defValues["train.decoder.activations"] = (None, "missing encoder activation")
		defValues["train.batch.size"] = (50, None)
		defValues["train.num.iterations"] = (500, None)
		defValues["train.loss.reduction"] = ("mean", None)
		defValues["train.lossFn"] = ("mse", None) 
		defValues["train.optimizer"] = ("adam", None) 
		defValues["train.opt.learning.rate"] = (.0001, None)
		defValues["train.opt.weight.decay"] = (0, None)
		defValues["train.opt.momentum"] = (0, None) 
		defValues["train.opt.eps"] = (1e-08, None) 
		defValues["train.opt.dampening"] = (0, None) 
		defValues["train.opt.momentum.nesterov"] = (False, None) 
		defValues["train.opt.betas"] = ([0.9, 0.999], None) 
		defValues["train.opt.alpha"] = (0.99, None) 
		defValues["train.noise.scale"] = (1.0, None)
		defValues["train.tied.weights"] = (False, None)
		defValues["train.model.save"] = (False, None)
		defValues["train.track.error"] = (False, None) 
		defValues["train.batch.intv"] = (5, None) 
		defValues["train.loss.av.window"] = (-1, None) 
		defValues["train.loss.diff.threshold"] = (0.05, None) 
		defValues["encode.use.saved.model"] = (True, None)
		defValues["encode.data.file"] = (None, "missing enoding data file")
		defValues["encode.feat.pad.size"] = (60, None)
		self.config = Configuration(configFile, defValues)

		super(AutoEncoder, self).__init__()

	# get config object
	def getConfig(self):
		return self.config
        
	def buildModel(self):
		"""
    	Loads configuration and builds the various piecess necessary for the model
		"""
		torch.manual_seed(9999)
		self.verbose = self.config.getStringConfig("common.verbose")[0]
		self.numinp = self.config.getIntConfig("train.num.input")[0]
		self.numHidden = self.config.getIntListConfig("train.num.hidden.units")[0]
		numLayers = len(self.numHidden)
		self.encActivations = self.config.getStringListConfig("train.encoder.activations")[0]
		self.decActivations = self.config.getStringListConfig("train.decoder.activations")[0]
		self.batchSize = self.config.getIntConfig("train.batch.size")[0]
		self.noiseScale = self.config.getFloatConfig("train.noise.scale")[0]
		self.numIter = self.config.getIntConfig("train.num.iterations")[0]
		self.optimizerStr = self.config.getStringConfig("train.optimizer")[0]
		self.optimizer = None
		self.loss = self.config.getStringConfig("train.lossFn")[0]
		self.tiedWeights = self.config.getBooleanConfig("train.tied.weights")[0]
		self.modelSave = self.config.getBooleanConfig("train.model.save")[0]
		self.useSavedModel = self.config.getBooleanConfig("encode.use.saved.model")[0]
		self.trackErr = self.config.getBooleanConfig("train.track.error")[0]
		self.batchIntv = self.config.getIntConfig("train.batch.intv")[0]
		self.restored = False
		
		#encoder
		inSize = self.numinp
		enModules = list()
		weights = list()
		for i in range(numLayers):
			outSize = self.numHidden[i]
			lin = nn.Linear(inSize, outSize)
			if self.tiedWeights:
				wt = torch.randn(outSize, inSize)
				weights.append(wt)
				lin.weight = nn.Parameter(wt)
			enModules.append(lin)
			act = self.encActivations[i]
			activation = FeedForwardNetwork.createActivation(act)
			if activation:
				enModules.append(activation)
			inSize = outSize
		self.encoder = nn.ModuleList(enModules)
			
        #decoder
		deModules = list()
		j = 0
		for i in reversed(range(numLayers)):
			inSize = self.numHidden[i]
			if i > 0:
				outSize = self.numHidden[i-1]
			else:
				outSize = self.numinp
			lin = nn.Linear(inSize, outSize)
			if self.tiedWeights:
				wt = weights[i]
				lin.weight = nn.Parameter(wt.transpose(0,1))
			deModules.append(lin)
			act = self.decActivations[j]
			j = j + 1
			activation = FeedForwardNetwork.createActivation(act)
			if activation:
				deModules.append(activation)
		self.decoder = nn.ModuleList(deModules)
		
		self.device = FeedForwardNetwork.getDevice(self)
		self.to(self.device)
		
	#deprecated
	def createActivation(self, act):
		"""
		create activation
		"""
		activation = None
		if act == "relu":
			activation = nn.ReLU()
		elif act == "sigmoid":
			activation = nn.Sigmoid()
		elif act == "noact":
			activation = None
		else:
			raise ValueError("invalid activation type")
	
	def forward(self, x):
		"""
		forward pass
		"""
		y = x
		for em in self.encoder:
			y = em(y)
		for dm in self.decoder:
			y = dm(y)
		return y

	#deprecated
	def prepTrainingData(self):
		"""
		loads and prepares training data
		"""
		#training data
		dataFile = self.config.getStringConfig("train.data.file")[0]
		featData = np.loadtxt(dataFile, delimiter=",",dtype=np.float32)
		if (self.config.getStringConfig("common.preprocessing")[0] == "scale"):
			featData = sk.preprocessing.scale(featData)
		#print(type(featData))
		return featData

	def getRegen(self):
		"""
		get regenerated data
		"""
		self.eval()
			
		teDataFile = self.config.getStringConfig("encode.data.file")[0]
		enData = FeedForwardNetwork.prepData(self, teDataFile, False)
		enData = torch.from_numpy(enData)
		enData = enData.to(self.device)
		with torch.no_grad():
			regenData = self(enData)
			regenData = regenData.data.cpu().numpy()
		data = (enData.data.cpu().numpy(), regenData)
		return data
		
	def regen(self):
		"""
		encode
		"""
		if (self.useSavedModel):
			# load saved model
			FeedForwardNetwork.restoreCheckpt(self)
		else:
			#train
			self.trainModel()
		self.eval()
			
		teDataFile = self.config.getStringConfig("encode.data.file")[0]
		enData = FeedForwardNetwork.prepData(self, teDataFile, False)
		enData = torch.from_numpy(enData)
		enData = enData.to(self.device)
		with torch.no_grad():
			regenData = self(enData)
			regenData = regenData.data.cpu().numpy()
		
		#score = perfMetric(self.loss, enData, regenData)
		#score = metrics.mean_squared_error(enData, regenData, multioutput="raw_values")
		#print(enData.shape, regenData.shape, score.shape)
		i = 0
		padWidth = self.config.getIntConfig("encode.feat.pad.size")[0]
		scores = list()
		for r in fileRecGen(teDataFile, ","):
			score = perfMetric(self.loss, enData[i:], regenData[i:])
			feat = (",".join(r)).ljust(padWidth, " ")
			rec = feat + "\t" + str(score)
			print(rec)
			i += 1
			fr = ",".join(r) + "," + str(score)
			scores.append(fr)
		return scores

		
	def getParams(self):
		"""
		get model parameters
		"""
		if (self.useSavedModel):
			# load saved model
			print ("...loading saved model")
			modelFilePath = self.getModelFilePath()
			self.load_state_dict(torch.load(modelFilePath))
			self.eval()
		else:
			self.trainAe()
	
	
		print("Model's state dict:")
		for paramTensor in self.state_dict():
			print(paramTensor)
			for te in self.state_dict()[paramTensor]:
				print(te)
				
	def getModelFilePath(self):
		""" 
		get model file path 
		"""
		modelDirectory = self.config.getStringConfig("common.model.directory")[0]
		modelFile = self.config.getStringConfig("common.model.file")[0]
		if modelFile is None:
			raise ValueError("missing model file name")
		modelFilePath = modelDirectory + "/" + modelFile
		return modelFilePath
		
	def trainModel(self):
		"""
		train model
		"""
		trDataFile = self.config.getStringConfig("train.data.file")[0]
		featData = FeedForwardNetwork.prepData(self, trDataFile, False)
		featData = torch.from_numpy(featData)
		self.featData = featData.to(self.device)
		self.dataloader = DataLoader(self.featData, batch_size=self.batchSize, shuffle=True)

		#if self.device == "cpu":
		#	model = self.cpu()
			
		# optimizer
		self.optimizer = FeedForwardNetwork.createOptimizer(self, self.optimizerStr)
			
		# loss function
		criterion = FeedForwardNetwork.createLossFunction(self, self.loss)
		
		# loss running average
		trLossAvWindowSz = self.config.getIntConfig("train.loss.av.window")[0]
		trLossDiffTh= self.config.getFloatConfig("train.loss.diff.threshold")[0]
		
		lossStat = SlidingWindowStat.createEmpty(trLossAvWindowSz) if trLossAvWindowSz > 0 else None
		peMean = None
		done = False
		for it in range(self.numIter):
			epochLoss = 0.0
			for data in self.dataloader:
				noisyData = data + self.noiseScale * torch.randn(*data.shape)
				noisyData = noisyData.to(self.device)
				output = self(noisyData)
				loss = criterion(output, data)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				epochLoss += loss.item()
			epochLoss /= len(self.dataloader)
			print('epoch [{}-{}], loss {:.6f}'.format(it + 1, self.numIter, epochLoss))
			
			if lossStat is not None:
				lossStat.add(epochLoss)
				if lossStat.getCurSize() == trLossAvWindowSz:
					(eMean, eSd) = lossStat.getStat()
					print("epoch loss average {:.6f}  std dev {:.6f}".format(eMean, eSd))
					if peMean is None:
						peMean = eMean
					else:
						ediff = abs(eMean - peMean)
						done =  ediff < (trLossDiffTh * eMean)
						peMean = eMean
			if done:
				print("traing loss converged")
				break
				
		self.evaluateModel()
		
		if self.modelSave:
			FeedForwardNetwork.saveCheckpt(self)

	def evaluateModel(self):
		"""
		evaluate model
		"""
		(enData, regenData) = self.getRegen()
		score = perfMetric(self.loss, enData, regenData)
		print("test error {:.6f}".format(score))
		
		
		