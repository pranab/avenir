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
from torch.nn import Linear
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
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
Graph convolution network
"""

class GraphConvoNetwork(nn.Module):
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
    	defValues["common.scaling.param.file"] = (None, None)
    	defValues["common.verbose"] = (False, None)
    	defValues["train.data.file"] = (None, "missing training data file")
    	defValues["train.data.num.nodes"] = (None, "missing num of nodes")
    	defValues["train.data.num.labeled"] = (None, "missing num of labeled nodes")
    	defValues["train.labeled.data.splits"] = ([75,15,10], None)
    	defValues["train.layer.data"] = (None, "missing layer data")
    	defValues["train.input.size"] = (None, "missing  output size")
    	defValues["train.output.size"] = (None, "missing  output size")
    	defValues["train.loss.reduction"] = ("mean", None)
    	defValues["train.num.iterations"] = (500, None)
    	defValues["train.lossFn"] = ("mse", None) 
    	defValues["train.optimizer"] = ("sgd", None)
    	defValues["train.opt.learning.rate"] = (.0001, None)
    	defValues["train.opt.weight.decay"] = (0, None)
    	defValues["train.opt.momentum"] = (0, None)
    	defValues["train.opt.eps"] = (1e-08, None)
    	defValues["train.opt.dampening"] = (0, None)
    	defValues["train.opt.momentum.nesterov"] = (False, None)
    	defValues["train.opt.betas"] = ([0.9, 0.999], None)
    	defValues["train.opt.alpha"] = (0.99, None)
    	defValues["train.save.model"] = (False, None)
    	defValues["train.track.error"] = (False, None)
    	defValues["train.epoch.intv"] = (5, None)
    	defValues["train.print.weights"] = (False, None)
    	defValues["valid.data.file"] = (None, None)
    	defValues["valid.accuracy.metric"] = (None, None)
    	self.config = Configuration(configFile, defValues)
    	super(GraphConvoNetwork, self).__init__()
    	
    
    def getConfig(self):
    	return self.config
    	
    def buildModel(self):
    	"""
    	Loads configuration and builds the various piecess necessary for the model
    	"""
    	torch.manual_seed(9999)
    	
    	self.verbose = self.config.getBooleanConfig("common.verbose")[0]
    	numinp = self.config.getIntConfig("train.input.size")[0]
    	self.outputSize = self.config.getIntConfig("train.output.size")[0]
    	self.numIter = self.config.getIntConfig("train.num.iterations")[0]
    	optimizer = self.config.getStringConfig("train.optimizer")[0]
    	self.lossFnStr = self.config.getStringConfig("train.lossFn")[0]
    	self.accMetric = self.config.getStringConfig("valid.accuracy.metric")[0]
    	self.trackErr = self.config.getBooleanConfig("train.track.error")[0]
    	self.restored = False
    	
    	#build network
    	layers = list()
    	ninp = numinp
    	trData =  self.config.getStringConfig("train.layer.data")[0].split(",")
    	for ld in trData:
    		lde = ld.split(":")
    		ne = len(lde)
    		assert ne == 5 or ne == 6, "expecting 5 or 6 items for layer data"
    		
    		gconv = False
    		if ne == 6:
    			if lde[0] == "gconv":
    				gconv == True
    			lde = lde[1:]
    		
    		#num of units, activation, whether batch normalize, whether batch normalize after activation, dropout fraction
    		nunit = int(lde[0])
    		actStr = lde[1]
    		act = FeedForwardNetwork.createActivation(actStr) if actStr != "none"  else None
    		bnorm = lde[2] == "true"
    		afterAct = lde[3] == "true"
    		dpr = float(lde[4])
    		
    		if gconv:
    			layers.append(GCNConv(ninp, nunit))
    		else:
    			layers.append(Linear(ninp, nunit))
    		if bnorm:
    			#with batch norm
    			if afterAct:
    				safeAppend(layers, act)
    				layers.append(torch.nn.BatchNorm1d(nunit))
    			else:
    				layers.append(torch.nn.BatchNorm1d(nunit))
    				safeAppend(layers, act)
    		else:
    			#without batch norm
    			safeAppend(layers, act)
    		
    		if dpr > 0:
    			layers.append(torch.nn.Dropout(dpr))
    		ninp = nunit
    		
    	self.layers = torch.nn.ModuleList(*layers)
    	self.loadData()
    	
    def loadData(self):
    	"""
    	load node and edge data
    	"""
    	dataFilePath = self.config.getStringConfig("train.data.file")[0]
    	numNodes = self.config.getIntConfig("train.data.num.nodes")[0]
    	numLabeled = self.config.getIntConfig("train.data.num.labeled")[0]
    	splits = self.config.getIntListConfig("train.labeled.data.splits")[0]
    	
    	dx = list()
    	dy = list()
    	edges = list()
    	for rec in fileRecGen(dataFilePath, delim):
    		if len(rec) > 2:
    			x = rec[1 :-1]
    			x = toFloatList(x)
    			y = int(rec[-1])
    			dx.append(x)
    			dy.append(y)
    		else:
    			e = toIntList(rec)
    			edges.append(e)
    			
    	dx = torch.tensor(dx, dtype=torch.float)
    	dy = torch.tensor(dy, dtype=torch.long)
    	edges = torch.tensor(edges, dtype=torch.long)
    	edges = edges.t().contiguous()
    	self.data = Data(x=dx, edge_index=edges)
    	
    	#maks
    	trStart = 0
    	vaStart = int(splits[0] * numLabeled)
    	teStart = vaStart + int(splits[1] * numLabeled)
    	
    	trMask = [False] * numNodes
    	trMask[0:vaStart] = [True] * vaStart
    	vaMask = [False] * numNodes
    	vaMask[vaStart:teStart] = [True] * (teStart - vaStart)
    	teMask = [False] * numNodes
    	teMask[teStart:] = [True] * (numNodes - teStart)
    	self.data.train_mask = torch.tensor(trMask, dtype=torch.bool)
    	self.data.val_mask = torch.tensor(vaMask, dtype=torch.bool)
    	self.data.test_mask = torch.tensor(teMask, dtype=torch.bool)
    	
    def descData(self):
    	"""
    	load node and edge data
    	"""
    	print(f'Number of nodes: {self.data.num_nodes}')
    	print(f'Number of edges: {self.data.num_edges}')
    	print(f'Number of node features: {self.data.num_node_features}')
    	print(f'Number of training nodes: {self.data.train_mask.sum()}')
    	print(f'Training node label rate: {int(self.data.train_mask.sum()) / data.num_nodes:.2f}')
    	print(f'Number of validation nodes: {self.data.val_mask.sum()}')
    	print(f'Number of test nodes: {self.data.test_mask.sum()}')
    	print(f'Is undirected: {self.data.is_undirected()}')
    	
    	print("Data attributes")
    	print(self.data.keys)
    	
    	print("Data types")
    	print(type(self.data.x))
    	print(type(self.data.y))
    	print(type(self.data.edge_index))
    	print(type(self.data.train_mask))
    	
    	print("Sample data")
    	print("x", self.data.x[:4])
    	print("y", self.data.y[:4])
    	print("edge", self.data.edge_index[:4])
    	print("train mask", self.data.train_mask[:4])
    	print("test mask", self.data.test_mask[:4])
    	
    	print("Any isolated node? " , self.data.has_isolated_nodes())
    	print("Any self loop? ", self.data.has_self_loops())
    	print("Is graph directed? ", self.data.is_directed())
    	
    def forward(self):
    	"""
    	
    	"""
    	x, edges = self.data.x, self.data.edge_index
    	for l in self.layers:
    		if isinstance(l, torch_geometric.nn.MessagePassing):
    			x = l(x, edges)
    		else:
    			x = l(x)
    		return x
    		
    @staticmethod
    def trainModel(model):
    	"""
    	train with batch data
    	"""
    	model.train()
    	if model.trackErr:
    		trErr = list()
    		vaErr = list()
    		
    	for epoch in range(model.numIter):
    		out = model()
    		loss = model.lossFn(out[self.data.train_mask], self.data.y[data.train_mask])
    		
    		#error tracking at batch level
    		if model.trackErr:
    			trErr.append(loss.item())
    			vErr = GraphConvoNetwork.evaluateModel(model)
    			vaErr.append(vErr)
    			
    		model.optimizer.zero_grad()
    		loss.backward()
    		model.optimizer.step()
    		
    @staticmethod
    def evaluateModel(model):
    	"""
    	evaluate model
    	"""
    	model.eval()
    	with torch.no_grad():
    		ypred = model(self.data)
    		ypred = ypred[data.test_mask].data.cpu().numpy()
    		yActual = self.data.y[data.test_mask].data.cpu().numpy()
    		score = perfMetric(model.lossFnStr, yActual, yPred)
    		
    	model.train()
    	return score
