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
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
"""
LSTM with one or more hidden layers with multi domensional data
"""

class LstmPredictor(nn.Module):
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
    	defValues["train.batch.size"] = (1, None)
    	defValues["train.batch.first"] = (False, None)
    	defValues["train.drop.prob"] = (0, None)
    	defValues["train.text.vocab.size"] = (-1, None)
    	defValues["train.text.embed.size"] = (-1, None)
    	defValues["train.optimizer"] = ("adam", None) 
    	defValues["train.learning.rate"] = (.0001, None)
    	defValues["train.betas"] = ("0.9, 0.999", None)
    	defValues["train.eps"] = (1e-8, None)
    	defValues["train.weight.decay"] = (.00001, None)
    	defValues["train.momentum"] = (0, None)
    	defValues["train.momentum.nesterov"] = (False, None)
    	defValues["train.out.sequence"] = (True, None)
    	defValues["train.out.activation"] = ("sigmoid", None)
    	defValues["train.loss"] = ("mse", None) 
    	defValues["train.grad.clip"] = (5, None) 
    	defValues["train.num.iterations"] = (500, None) 

    	self.config = Configuration(configFile, defValues)
  
    	super(LstmClassify, self).__init__()
    	
    def getConfig(self):
    	return self.config
    	
    def buildModel(self):
    	"""
    	Loads configuration and builds the various piecess necessary for the model
    	"""
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
    	self.dropout = nn.Dropout(dropProb)
    	self.linear = nn.Linear(self.hiddenSize, self.outputSize)
    	outAct = self.config.getStringConfig("train.out.activation")[0]
    	if outAct == "sigmoid":
    		self.outAct = nn.Sigmoid()
    	elif outAct == "softmax":
    		self.outAct = nn.Softmax(dim=1)
    	else:
    		self.outAct = None
    		
    	
    	#load data
    	dataFilePath = self.config.getStringConfig("train.data.file")[0]
    	fCols = self.config.getIntListConfig("train.data.feat.cols")[0]
    	tCol = self.config.getIntConfig("train.data.target.col")[0]
    	delim = self.config.getStringConfig("train.data.delim")[0]
    	self.fData, self.tData = self.loadData(dataFilePath, delim, fCols[0], fCols[1], tCol)
    	
    	self.batchSize = self.config.getIntConfig("train.batch.size")[0]
    	self.dataSize = self.fData.size()[0]
    	self.numBatch = int(self.dataSize / self.batchSize)
    	
    def loadData(self, filePath, delim, scolStart, scolEnd, targetCol):
    	"""
    	loads data for file with one sequence per line and data can be a vector
    	"""
    	cols = list(range(scolStart, scolEnd, 1))
    	cols.append(targetCol)
    	data = np.loadtxt(file, delimiter=delim, usecols=cols)
    	sData = data[:, :-1]
    	tData = data[:, -1]
    	return (SData.astype(np.float32), tData.astype(np.float32))
    	
    def batchTransform(self):
    	"""
    	transforms data from (dataSize, seqLength x inputSize) to (batch, seqLength, inputSize) tensor
    	or (seqLength, batch, inputSize) tensor
    	"""
    	if self.batchFirst:
    		bfData = torch.zeros(self.batchSize, self.seqLen, self.inputSize)
    	else:
    		bfData = torch.zeros(self.seqLen, self.batchSize, self.inputSize)
    	
    	btData = torch.zeros(self.batchSize,self.numBatch)
    	
    	for bi in range(self.numBatch):
    		for bdi in range(self.batchSize):
    			di = sampleUniform(0, self.dataSize-1)
    			row = self.fData[di]
    			for ci, cv in enumarate(row):
    				si = int(ci / self.seqLen)
    				ii = ci % self.seqLen
    				if self.batchFirst:
    					bfData[bi][si][ii] = cv
    				else:
    					bfData[si][bi][ii] = cv
    			btData[bi][bdi] = self.tData[di]
    			
    	return (bfData, btData)		
		
	
    def forward(self, x, h):
    	"""
    	Forward pass
    	"""
    	lstmOut, hout = self.lstm(x,h)
    	if self.outSeq:
    		pass
    	else:
    		lstmOut = lstmOut[self.seqLen - 1].view(-1, self.hiddenSize)
    		out = self.dropout(lstmOut)
    		out = self.linear(out)
    		if self.outAct:
    			out = self.outAct(out)
    		out = out.view(self.batchSize, -1)
    		out = out[:,-1]
    		return out, hout
    
    def initHidden(self):
    	"""
    	Initialize hidden weights
    	"""
    	hidden = (torch.zeros(self.nLayers,self.batchSize,self.hiddenSize),
    	torch.zeros(self.nLayers,self.batchSize,self.hiddenSize))
    	return hidden 
                 
    def createOptomizer(self):
    	"""
    	Create optimizer
    	"""
    	optimizerName = self.config.getStringConfig("train.optimizer")[0]
    	learnRate = self.config.getFloatConfig("train.learning.rate")[0]
    	weightDecay = self.config.getFloatConfig("train.weight.decay")[0]
    	if optimizerName == "adam":
    		betas = self.config.getStringConfig("train.betas")[0]
    		betas = strToFloatArray(self.betas, ",")
    		betas = (betas[0], betas[1]) 
    		eps = self.config.getFloatConfig("train.eps")[0]
    		optimizer = torch.optim.Adam(self.parameters(), lr=learnRate, betas=betas, eps = eps,
    		weight_decay=weightDecay)
    	elif optimizerName == "sgd":
    		momentum = self.config.getFloatConfig("train.momentum")[0]
    		dampening = self.config.getFloatConfig("train.dampening")[0]
    		momentumNesterov = self.config.getBooleanConfig("train.momentum.nesterov")[0]
    		optimizer = torch.optim.SGD(self.parameters(), momentum=momentum, dampening=dampening,
    		weight_decay=weightDecay, nesterov=momentumNesterov)
    	else:
    		raise ValueError("invalid optimizer type")
    	return optimizer
    	
    def createLossFun(self):
    	"""
    	Create loss function
    	"""
    	loss = self.config.getStringConfig("train.loss")[0]
    	if loss == "mse":
    		criterion = nn.MSELoss()
    	elif loss == "ce":
    		criterion = nn.CrossEntropyLoss()
    	elif loss == "bce":
    		criterion = nn.BCELoss()
    	else:
    		raise ValueError("invalid loss function")
    	return criterion
    	
    def trainLstm(self):
    	"""
    	train lstm
    	"""
    	self.train()
    	device = self.config.getStringConfig("common.device")[0]
    	self.to(device)
    	optimizer = self.createOptomizer()
    	criterion = self.createLossFun()
    	clip = self.config.getFloatConfig("train.grad.clip")[0]
    	numIter = self.config.getIntConfig("train.num.iterations")[0]
    	
    	for it in range(numIter):
    		#forward pass
    		hid = model.initHidden()
    		inputs, labels = self.batchTransform()
    		inputs, labels = inputs.to(device), labels.to(device)
    		output, hid = self(inputs, hid)
    		loss = criterion(output.squeeze(), labels)
    		if self.verbose and it % 50 == 0:
    			print("epoch {}  loss {:.6f}".format(t, loss.item()))
    		
    		# Zero gradients, perform a backward pass, and update the weights.
    		optimizer.zero_grad()
    		loss.backward()
    		nn.utils.clip_grad_norm_(model.parameters(), clip)
    		optimizer.step()
			
		
