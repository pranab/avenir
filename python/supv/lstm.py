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
from tnn import FeedForwardNetwork

"""
LSTM with one or more hidden layers with multi domensional data
"""

class LstmNetwork(nn.Module):
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
    	defValues["common.verbose"] = (False, None)
    	defValues["common.device"] = ("cpu", None)
    	defValues["train.data.file"] = (None, "missing training data file path")
    	defValues["train.data.type"] = ("numeric", None)
    	defValues["train.data.feat.cols"] = (None, "missing feature columns")
    	defValues["train.data.target.col"] = (None, "missing target column")
    	defValues["train.data.delim"] = (",", None)
    	defValues["train.input.size"] = (None, "missing  input size")
    	defValues["train.hidden.size"] = (None, "missing  hidden size")
    	defValues["train.output.size"] = (None, "missing  output size")
    	defValues["train.num.layers"] = (1, None)
    	defValues["train.seq.len"] = (1, None)
    	defValues["train.batch.size"] = (32, None)
    	defValues["train.batch.first"] = (False, None)
    	defValues["train.drop.prob"] = (0, None)
    	defValues["train.optimizer"] = ("adam", None)
    	defValues["train.opt.learning.rate"] = (.0001, None)
    	defValues["train.opt.weight.decay"] = (0, None)
    	defValues["train.opt.momentum"] = (0, None)
    	defValues["train.opt.eps"] = (1e-08, None)
    	defValues["train.opt.dampening"] = (0, None)
    	defValues["train.opt.momentum.nesterov"] = (False, None)
    	defValues["train.opt.betas"] = ([0.9, 0.999], None)
    	defValues["train.opt.alpha"] = (0.99, None)
    	defValues["train.out.sequence"] = (True, None)
    	defValues["train.out.activation"] = ("sigmoid", None)
    	defValues["train.loss.fn"] = ("mse", None) 
    	defValues["train.loss.reduction"] = ("mean", None)
    	defValues["train.grad.clip"] = (5, None) 
    	defValues["train.num.iterations"] = (500, None)
    	defValues["train.save.model"] = (False, None) 
    	defValues["valid.data.file"] = (None, "missing validation data file path")
    	defValues["valid.accuracy.metric"] = (None, None)
    	defValues["predict.data.file"] = (None, None)
    	defValues["predict.use.saved.model"] = (True, None)
    	defValues["predict.output"] = ("binary", None)
    	defValues["predict.feat.pad.size"] = (60, None)

    	self.config = Configuration(configFile, defValues)
  
    	super(LstmNetwork, self).__init__()
    	
    def getConfig(self):
    	return self.config
    	
    def buildModel(self):
    	"""
    	Loads configuration and builds the various piecess necessary for the model
    	"""
    	torch.manual_seed(9999)
    	self.verbose = self.config.getStringConfig("common.verbose")[0]
    	self.inputSize = self.config.getIntConfig("train.input.size")[0]
    	self.outputSize = self.config.getIntConfig("train.output.size")[0]
    	self.nLayers = self.config.getIntConfig("train.num.layers")[0]
    	self.hiddenSize = self.config.getIntConfig("train.hidden.size")[0]
    	self.seqLen = self.config.getIntConfig("train.seq.len")[0]
    	self.batchSize = self.config.getIntConfig("train.batch.size")[0]
    	self.batchFirst = self.config.getBooleanConfig("train.batch.first")[0]
    	dropProb = self.config.getFloatConfig("train.drop.prob")[0]
    	self.outSeq = self.config.getBooleanConfig("train.out.sequence")[0]
    	
    	#model
    	self.lstm = nn.LSTM(self.inputSize, self.hiddenSize, self.nLayers, dropout=dropProb, batch_first=self.batchFirst)
    	self.linear = nn.Linear(self.hiddenSize, self.outputSize)
    	outAct = self.config.getStringConfig("train.out.activation")[0]
    	self.outAct = FeedForwardNetwork.createActivation(outAct)   		
    	
    	#load training data
    	dataFilePath = self.config.getStringConfig("train.data.file")[0]
    	self.fCols = self.config.getIntListConfig("train.data.feat.cols")[0]
    	assert len(self.fCols) == 2, "specify only start and end columns of features"
    	self.tCol = self.config.getIntConfig("train.data.target.col")[0]
    	self.delim = self.config.getStringConfig("train.data.delim")[0]
    	
    	self.fData, self.tData = self.loadData(dataFilePath, self.delim, self.fCols[0],self.fCols[1], self.tCol)
    	self.fData = torch.from_numpy(self.fData)
    	self.tData = torch.from_numpy(self.tData)
    	
    	#load validation data
    	vaDataFilePath = self.config.getStringConfig("valid.data.file")[0]
    	self.vfData, self.vtData = self.loadData(vaDataFilePath, self.delim, self.fCols[0], self.fCols[1], self.tCol)
    	self.vfData = torch.from_numpy(self.vfData)
    	self.vtData = torch.from_numpy(self.vtData)
    	
    	self.batchSize = self.config.getIntConfig("train.batch.size")[0]
    	self.dataSize = self.fData.shape[0]
    	self.numBatch = int(self.dataSize / self.batchSize)
    	
    def loadData(self, filePath, delim, scolStart, scolEnd, targetCol):
    	"""
    	loads data for file with one sequence per line and data can be a vector
    	"""
    	if targetCol >= 0:
    		#include target column
    		cols = list(range(scolStart, scolEnd + 1, 1))
    		cols.append(targetCol)
    		data = np.loadtxt(filePath, delimiter=delim, usecols=cols)
    		#one output for whole sequence
    		sData = data[:, :-1]
    		if (self.config.getStringConfig("common.preprocessing")[0] == "scale"):
    			sData = self.scaleSeqData(sData)
    		tData = data[:, -1]
    	
    		#target int (index into class labels)  for classification 
    		sData = sData.astype(np.float32)
    		tData = tData.astype(np.float32) if self.outputSize == 1 else tData.astype(np.long)
    		exData =  (sData, tData)
    	else:
    		#exclude target column
    		cols = list(range(scolStart, scolEnd + 1, 1))
    		data = np.loadtxt(filePath, delimiter=delim, usecols=cols)

    		#one output for whole sequence
    		sData = data
    		if (self.config.getStringConfig("common.preprocessing")[0] == "scale"):
    			sData = self.scaleSeqData(sData)
    	
    		#target int (index into class labels)  for classification 
    		sData = sData.astype(np.float32)
    		exData =  sData
    	
    	return exData
    	
    def scaleSeqData(self, sData):
    	"""
    	scales data transforming non squence format
    	"""
    	scalingMethod = self.config.getStringConfig("common.scaling.method")[0]
    	sData = fromMultDimSeqToTabular(sData, self.inputSize, self.seqLen)
    	sData = scaleData(sData, scalingMethod)
    	sData = fromTabularToMultDimSeq(sData, self.inputSize, self.seqLen)
    	return sData
    		
    def formattedBatchGenarator(self):
    	"""
    	transforms traing data from (dataSize, seqLength x inputSize) to (batch, seqLength, inputSize) tensor
    	or (seqLength, batch, inputSize) tensor
    	"""
    	
    	for _ in range(self.numBatch):
    		bfData = torch.zeros([self.batchSize, self.seqLen, self.inputSize], dtype=torch.float32) if self.batchFirst\
    		else torch.zeros([self.seqLen, self.batchSize, self.inputSize], dtype=torch.float32)
    		tdType = torch.float32 if self.outputSize == 1 else torch.long
    		btData = torch.zeros([self.batchSize], dtype=tdType)
    		
    		i = 0
    		for bdi in range(self.batchSize):
    			di = sampleUniform(0, self.dataSize-1)
    			row = self.fData[di]
    			for ci, cv in enumerate(row):
    				si = int(ci / self.inputSize)
    				ii = ci % self.inputSize
    				if self.batchFirst:
    					bfData[bdi][si][ii] = cv
    				else:
    					#print(si, bdi, ii)
    					bfData[si][bdi][ii] = cv
    			btData[i] = self.tData[di]
    			i += 1
    		
    		#for seq output correct first 2 dimensions
    		if  self.outSeq and not self.batchFirst:
    			btData = torch.transpose(btData,0,1)
 
    		yield (bfData, btData)		
		
    def formatData(self, fData, tData=None):
    	"""
    	transforms validation or prediction data data from (dataSize, seqLength x inputSize) to 
    	(batch, seqLength, inputSize) tensor or (seqLength, batch, inputSize) tensor
    	"""
    	dSize = fData.shape[0]
    	bfData = torch.zeros([dSize, self.seqLen, self.inputSize], dtype=torch.float32) if self.batchFirst\
    	else torch.zeros([self.seqLen, dSize, self.inputSize], dtype=torch.float32)
    	
    	for ri in range(dSize):    
    		row = fData[ri]
    		for ci, cv in enumerate(row):
    			si = int(ci / self.inputSize)
    			ii = ci % self.inputSize
    			if self.batchFirst:
    				bfData[ri][si][ii] = cv
    			else:
    				bfData[si][ri][ii] = cv
    	if tData is not None:
    		btData = torch.transpose(tData,0,1) if  self.outSeq and not self.batchFirst else tData
    		formData =  (bfData, btData)
    	else:
    		formData  = bfData
    	return formData
		
    def forward(self, x, h):
    	"""
    	Forward pass
    	"""
    	out, hout = self.lstm(x,h)
    	if self.outSeq:
    		# seq to seq prediction
    		out = out.view(-1, self.hiddenSize)
    		out = self.linear(out)
    		if self.outAct is not None:
    			out = self.outAct(out)
    		out = out.view(self.batchSize * self.seqLen, -1)
    	else:
    		#seq to one prediction
    		out = out[self.seqLen - 1].view(-1, self.hiddenSize)
    		out = self.linear(out)
    		if self.outAct is not None:
    			out = self.outAct(out)
    		#out = out.view(self.batchSize, -1)
    		
    	return out, hout
    
    def initHidden(self, batch):
    	"""
    	Initialize hidden weights
    	"""
    	hidden = (torch.zeros(self.nLayers,batch,self.hiddenSize),
    	torch.zeros(self.nLayers,batch,self.hiddenSize))
    	return hidden 
                     	
    def trainLstm(self):
    	"""
    	train lstm
    	"""
    	print("..starting training")
    	self.train()
 
    	device = self.config.getStringConfig("common.device")[0]
    	self.to(device)
    	optimizerName = self.config.getStringConfig("train.optimizer")[0]
    	self.optimizer = FeedForwardNetwork.createOptimizer(self, optimizerName)
    	lossFn = self.config.getStringConfig("train.loss.fn")[0]
    	criterion = FeedForwardNetwork.createLossFunction(self, lossFn)
    	clip = self.config.getFloatConfig("train.grad.clip")[0]
    	numIter = self.config.getIntConfig("train.num.iterations")[0]
    	accMetric = self.config.getStringConfig("valid.accuracy.metric")[0]
    	
 
    	for it in range(numIter):
    		b = 0
    		for inputs, labels in self.formattedBatchGenarator():
    			#forward pass
    			hid = self.initHidden(self.batchSize)
    			inputs, labels = inputs.to(device), labels.to(device)
    			output, hid = self(inputs, hid)
    			
    			#loss
    			if self.outSeq:
    				labels = labels.view(self.batchSize * self.seqLen, -1)
    			loss = criterion(output, labels)
    			
    			if self.verbose and it % 50 == 0 and b % 10 == 0:
    				print("epoch {}  batch {}  loss {:.6f}".format(it, b, loss.item()))
    		
    			# zero gradients, perform a backward pass, and update the weights.
    			self.optimizer.zero_grad()
    			loss.backward()
    			nn.utils.clip_grad_norm_(self.parameters(), clip)
    			self.optimizer.step()
    			b += 1
    	
    	#validate		
    	print("..validating model")
    	self.eval()
    	with torch.no_grad():
    		fData, tData = self.formatData(self.vfData, self.vtData)
    		fData = fData.to(device)
    		vsize = tData.shape[0]
    		hid = self.initHidden(vsize)
    		yPred, _ = self(fData, hid)
    		yPred = yPred.data.cpu().numpy()
    		yActual = tData.data.cpu().numpy()
    	
    	if self.verbose:
    		print("\npredicted \t\t actual")
    		for i in range(vsize):
    			print(str(yPred[i]) + "\t" + str(yActual[i]))
    		
    	score = perfMetric(accMetric, yActual, yPred)
    	print(formatFloat(3, score, "perf score"))
    	
    	#save
    	modelSave = self.config.getBooleanConfig("train.model.save")[0]
    	if modelSave:
    		FeedForwardNetwork.saveCheckpt(self)
    		
    def predictLstm(self):
    	"""
    	predict
    	"""
    	print("..predicting using model")
    	useSavedModel = self.config.getBooleanConfig("predict.use.saved.model")[0]
    	if useSavedModel:
    		FeedForwardNetwork.restoreCheckpt(self)
    	else:
    		self.trainLstm()
    		
    	prDataFilePath = self.config.getStringConfig("predict.data.file")[0]
    	pfData = self.loadData(prDataFilePath, self.delim, self.fCols[0], self.fCols[1], -1)
    	pfData = torch.from_numpy(pfData)
    	dsize = pfData.shape[0]
    	
    	#predict
    	device = self.config.getStringConfig("common.device")[0]
    	self.eval()
    	with torch.no_grad():
    		fData = self.formatData(pfData)
    		fData = fData.to(device)
    		hid = self.initHidden(dsize)
    		yPred, _ = self(fData, hid)
    		yPred = yPred.data.cpu().numpy()
    		
    	if self.outputSize == 2:
    		#classification
    		yPred = FeedForwardNetwork.processClassifOutput(yPred, self.config)
    	
    	# print prediction
    	FeedForwardNetwork.printPrediction(yPred, self.config)

			

		
	
		
