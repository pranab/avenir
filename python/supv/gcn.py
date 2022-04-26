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
import matplotlib
import random
from random import randint
from itertools import compress
import numpy as np
import torch
from torch import nn
from torch.nn import Linear
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import sklearn as sk
import jprops
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
    	initilizer
    	
		Parameters
			configFile : config file path
    	"""
    	defValues = dict()
    	defValues["common.model.directory"] = ("model", None)
    	defValues["common.model.file"] = (None, None)
    	defValues["common.preprocessing"] = (None, None)
    	defValues["common.scaling.method"] = ("zscale", None)
    	defValues["common.scaling.minrows"] = (50, None)
    	defValues["common.scaling.param.file"] = (None, None)
    	defValues["common.verbose"] = (False, None)
    	defValues["common.device"] = ("cpu", None)
    	defValues["train.data.file"] = (None, "missing training data file")
    	defValues["train.data.num.nodes.total"] = (None, None)
    	defValues["train.data.num.nodes.training"] = (None, None)
    	defValues["train.data.splits"] = ([.75,.15,.10], None)
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
    	defValues["valid.accuracy.metric"] = (None, None)
    	defValues["predict.create.mask"] = (False, None)
    	defValues["predict.use.saved.model"] = (True, None)
    	
    	self.config = Configuration(configFile, defValues)
    	super(GraphConvoNetwork, self).__init__()
    	
    
    def getConfig(self):
    	"""
    	return config
    	"""
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
    	self.clabels = list(range(self.outputSize)) if self.outputSize > 1 else None
    	
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
    		
    	self.layers = torch.nn.ModuleList(layers)
    	self.device = FeedForwardNetwork.getDevice(self)
    	self.to(self.device)
    	self.loadData()
    	
    	self.lossFn = FeedForwardNetwork.createLossFunction(self, self.lossFnStr)
    	self.optimizer =  FeedForwardNetwork.createOptimizer(self, optimizer)
    	self.trained = False
    	
    def loadData(self):
    	"""
    	load node and edge data
    	"""
    	dataFilePath = self.config.getStringConfig("train.data.file")[0]
    	numNodes = self.config.getIntConfig("train.data.num.nodes.total")[0]
    	numLabeled = self.config.getIntConfig("train.data.num.nodes.training")[0]
    	splits = self.config.getFloatListConfig("train.data.splits")[0]
    	crPredMask = self.config.getBooleanConfig("predict.create.mask")[0]
    	
    	dx = list()
    	dy = list()
    	edges = list()
    	mask = None
    	for rec in fileRecGen(dataFilePath, ","):
    		if len(rec) > 2:
    			x = rec[1 :-1]
    			x = toFloatList(x)
    			y = int(rec[-1])
    			dx.append(x)
    			dy.append(y)
    		elif len(rec) == 2:
    			e = toIntList(rec)
    			edges.append(e)
    		elif len(rec) == 1:
    			items = rec[0].split()
    			assertEqual(items[0], "mask", "invalid mask data")
    			numNodes = int(items[1])
    			print(numNodes)
    			mask = list()
    			for r in range(2, len(items), 1):
    				ri = items[r].split(":")
    				#print(ri)
    				ms = list(range(int(ri[0]), int(ri[1]), 1))
    				mask.extend(ms)
    	#scale node features
    	if (self.config.getStringConfig("common.preprocessing")[0] == "scale"):
    		scalingMethod = self.config.getStringConfig("common.scaling.method")[0]
    		dx = scaleData(dx, scalingMethod)
				
    	dx = torch.tensor(dx, dtype=torch.float)
    	dy = torch.tensor(dy, dtype=torch.long)
    	edges = torch.tensor(edges, dtype=torch.long)
    	edges = edges.t().contiguous()
    	dx = dx.to(self.device)
    	dy = dy.to(self.device)
    	edges = edges.to(self.device)
    	self.data = Data(x=dx, edge_index=edges, y=dy)
    	
    	#maks
    	if mask is None:
    		#trainiug data in the beginning
    		trStart = 0
    		vaStart = int(splits[0] * numLabeled)
    		teStart = vaStart + int(splits[1] * numLabeled)

    		trMask = [False] * numNodes
    		trMask[0:vaStart] = [True] * vaStart
    		vaMask = [False] * numNodes
    		vaMask[vaStart:teStart] = [True] * (teStart - vaStart)
    		teMask = [False] * numNodes
    		teMask[teStart:] = [True] * (numNodes - teStart)
    	else:
    		#training data anywhere
    		if crPredMask:
    			prMask = [True] * numNodes
    			for i in mask:
    				prMask[i] = False
    		self.prMask = torch.tensor(prMask, dtype=torch.bool)
    		
    		nshuffle = int(len(mask) / 2)
    		shuffle(mask, nshuffle)
    		#print(mask)
    		lmask = len(mask)
    		trme = int(splits[0] * lmask)
    		vame = int((splits[0] + splits[1]) * lmask)
    		teme = lmask
    		trMask = [False] * numNodes
    		for i in mask[:trme]:
    			trMask[i] = True
    		vaMask = [False] * numNodes
    		for i in mask[trme:vame]:
    			vaMask[i] = True
    		teMask = [False] * numNodes
    		for i in mask[vame:]:
    			teMask[i] = True
    		#print(vaMask)
    	
    	trMask = torch.tensor(trMask, dtype=torch.bool)
    	trMask = trMask.to(self.device)
    	self.data.train_mask = trMask
    	vaMask = torch.tensor(vaMask, dtype=torch.bool)
    	vaMask = vaMask.to(self.device)
    	self.data.val_mask = vaMask
    	teMask = torch.tensor(teMask, dtype=torch.bool)
    	teMask = teMask.to(self.device)
    	self.data.test_mask = teMask
    	
    		
    def descData(self):
    	"""
    	describe data
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
    	forward prop
    	"""
    	x, edges = self.data.x, self.data.edge_index
    	for l in self.layers:
    		if isinstance(l, MessagePassing):
    			x = l(x, edges)
    		else:
    			x = l(x)
    	return x
    		
    @staticmethod
    def trainModel(model):
    	"""
    	train with batch data
    	
		Parameters
			model : torch model
    	"""
    	epochIntv = model.config.getIntConfig("train.epoch.intv")[0]
    	
    	model.train()
    	if model.trackErr:
    		trErr = list()
    		vaErr = list()
    		
    	for epoch in range(model.numIter):
    		out = model()
    		loss = model.lossFn(out[model.data.train_mask], model.data.y[model.data.train_mask])
    	
    		#error tracking at batch level
    		if model.trackErr:
    			trErr.append(loss.item())
    			vErr = GraphConvoNetwork.evaluateModel(model)
    			vaErr.append(vErr)
    			if model.verbose and epoch % epochIntv == 0:
    				print("epoch {}   loss {:.6f}  val error {:.6f}".format(epoch, loss.item(), vErr))
    			
    		model.optimizer.zero_grad()
    		loss.backward()
    		model.optimizer.step()
    	
    	#acc = GraphConvoNetwork.evaluateModel(model, True)	
    	#print(acc)
    	modelSave = model.config.getBooleanConfig("train.model.save")[0]
    	if modelSave:
    		FeedForwardNetwork.saveCheckpt(model)
    		
    	if model.trackErr:
    		FeedForwardNetwork.errorPlot(model, trErr, vaErr)
    		
    	model.trained = True	
  	
    @staticmethod
    def evaluateModel(model, verbose=False):
    	"""
    	evaluate model
    	
		Parameters
			model : torch model
			verbose : if True additional output
    	"""
    	model.eval()
    	with torch.no_grad():
    		out = model()
    		if verbose:
    			print(out)
    		yPred = out[model.data.val_mask].data.cpu().numpy()
    		yActual = model.data.y[model.data.val_mask].data.cpu().numpy()
    		if verbose:
    			for pa in zip(yPred, yActual):
    				print(pa)
    		#correct = yPred == yActual
    		#score = int(correct.sum()) / int(model.data.val_mask.sum())
    		
    		score = perfMetric(model.lossFnStr, yActual, yPred, model.clabels)
    		
    	model.train()
    	return score
    	
    @staticmethod
    def validateModel(model, retPred=False):
    	"""
		model validation
		
		Parameters
			model : torch model
			retPred : if True return prediction
		"""
    	model.eval()
    	with torch.no_grad():
    		out = model()
    		yPred = out.argmax(dim=1)
    		yPred = yPred[model.data.test_mask].data.cpu().numpy()
    		yActual = model.data.y[model.data.test_mask].data.cpu().numpy()
    		#correct = yPred == yActual
    		#score = int(correct.sum()) / int(model.data.val_mask.sum())
    		score = perfMetric(model.accMetric, yActual, yPred)
    		print(formatFloat(3, score, "test #perf score"))
    	return score
    	
    @staticmethod
    def modelPrediction(model, inclData=True):
    	"""
    	make prediction
    	
		Parameters
			model : torch model
    		inclData : True to include input data
    	"""
    	cmask = model.config.getBooleanConfig("predict.create.mask")[0]
    	if not cmask:
    		print("create prediction mask property needs to be set to True")
    		return None
    		
    	useSavedModel = model.config.getBooleanConfig("predict.use.saved.model")[0]
    	if useSavedModel:
    		FeedForwardNetwork.restoreCheckpt(model)
    	else:
    		if not model.trained:
    			GraphConvoNetwork.trainModel(model)
    	
    	model.eval()
    	with torch.no_grad():
    		out = model()
    		yPred = out.argmax(dim=1)
    		yPred = yPred[model.prMask].data.cpu().numpy()
    	
    	if inclData:
    		dataFilePath = model.config.getStringConfig("train.data.file")[0]	
    		filt = lambda r : len(r) > 2
    		ndata = list(fileFiltRecGen(dataFilePath, filt))
    		prMask = model.prMask.data.cpu().numpy()
    		assertEqual(len(ndata), prMask.shape[0], "data and mask lengths are not equal")
    		precs = list(compress(ndata, prMask))
    		precs = list(map(lambda r : r[:-1], precs))
    		assertEqual(len(precs), yPred.shape[0], "data and mask lengths are not equal")
    		res =  zip(precs, yPred)
    	else:
    		res = yPred
    	return res
    	
