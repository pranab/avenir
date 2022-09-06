#!/usr/local/bin/python3

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
import random
import jprops
from random import randint
from matumizi.util import *
from matumizi.mlutil import *

"""
Markov chain classifier
"""
class MarkovChainClassifier():
	def __init__(self, configFile):
		"""
		constructor
		
		Parameters
			configFile: config file path
		"""
		defValues = {}
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["common.states"] = (None, "missing state list")
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.class.labels"] = (["F", "T"], None)
		defValues["train.data.key.len"] = (1, None)
		defValues["train.model.save"] = (False, None)
		defValues["train.score.method"] = ("accuracy", None)
		defValues["predict.data.file"] = (None, None)
		defValues["predict.use.saved.model"] = (True, None)
		defValues["predict.log.odds.threshold"] = (0, None)
		defValues["validate.data.file"] = (None, "missing validation data file")
		defValues["validate.use.saved.model"] = (False, None)
		defValues["valid.accuracy.metric"] = ("acc", None)
		self.config = Configuration(configFile, defValues)
		
		self.stTranPr = dict()
		self.clabels = self.config.getStringListConfig("train.data.class.labels")[0]
		self.states = self.config.getStringListConfig("common.states")[0]
		self.nstates = len(self.states)
		for cl in self.clabels:
			stp = np.ones((self.nstates,self.nstates))
			self.stTranPr[cl] = stp
		
	def train(self):
		"""
		train model
		"""	
		#state transition matrix
		tdfPath = self.config.getStringConfig("train.data.file")[0]
		klen = self.config.getIntConfig("train.data.key.len")[0]
		for rec in fileRecGen(tdfPath):
			cl = rec[klen]
			rlen = len(rec)
			for i in range(klen+1, rlen-1, 1):
				fst = self.states.index(rec[i])
				tst = self.states.index(rec[i+1])
				self.stTranPr[cl][fst][tst] += 1
		
		#normalize to probability
		for cl in self.clabels:
			stp = self.stTranPr[cl]
			for i in range(self.nstates):
				s = stp[i].sum()
				r = stp[i] / s
				stp[i] = r
		
		#save		
		if 	self.config.getBooleanConfig("train.model.save")[0]:
			mdPath = self.config.getStringConfig("common.model.directory")[0]
			assert os.path.exists(mdPath), "model save directory does not exist"
			mfPath = self.config.getStringConfig("common.model.file")[0]
			mfPath = os.path.join(mdPath, mfPath)

			with open(mfPath, "w") as fh:
				for cl in self.clabels:
					fh.write("label:" + cl +"\n")
					stp = self.stTranPr[cl]
					for r in stp:
						rs = ",".join(toStrList(r, 6)) + "\n"
						fh.write(rs)

	def validate(self):
		"""
		validate using  model
		"""	
		useSavedModel = self.config.getBooleanConfig("predict.use.saved.model")[0]
		if useSavedModel:
			self.__restoreModel()
		else:
			self.train() 
			
		vdfPath = self.config.getStringConfig("validate.data.file")[0]	
		accMetric = self.config.getStringConfig("valid.accuracy.metric")[0]
		
		yac, ypr = self.__getPrediction(vdfPath, True)
		if type(self.clabels[0]) == str:
			yac = self.__toIntClabel(yac)
			ypr = self.__toIntClabel(ypr)
		score = perfMetric(accMetric, yac, ypr)
		print(formatFloat(3, score, "perf score"))

			
	def predict(self):
		"""
		predict using  model
		"""	
		useSavedModel = self.config.getBooleanConfig("predict.use.saved.model")[0]
		if useSavedModel:
			self.__restoreModel()
		else:
			self.train() 
			
		#predict
		pdfPath = self.config.getStringConfig("predict.data.file")[0]
		_ , ypr = self.__getPrediction(pdfPath)
		return ypr
		
	def __restoreModel(self):
		"""
		restore model
		"""
		mdPath = self.config.getStringConfig("common.model.directory")[0]
		assert os.path.exists(mdPath), "model save directory does not exist"
		mfPath = self.config.getStringConfig("common.model.file")[0]
		mfPath = os.path.join(mdPath, mfPath)
		stp = None
		cl = None
		for rec in fileRecGen(mfPath):
			if len(rec) == 1:
				if stp is not None:
					stp = np.array(stp)
					self.stTranPr[cl] = stp
				cl = rec[0].split(":")[1]
				stp = list()
			else:
				frec = asFloatList(rec)
				stp.append(frec)
				
		stp = np.array(stp)
		self.stTranPr[cl] = stp
				
	def __getPrediction(self, fpath, validate=False):
		"""
		get predictions
		
		Parameters
			fpath : data file path
			validate: True if validation
		"""
	
		nc = self.clabels[0]
		pc = self.clabels[1]
		thold = self.config.getFloatConfig("predict.log.odds.threshold")[0]
		klen = self.config.getIntConfig("train.data.key.len")[0]
		offset = klen+1 if validate else klen
		ypr = list()
		yac = list()
		for rec in fileRecGen(fpath):
			lodds = 0
			rlen = len(rec)
			for i in range(offset, rlen-1, 1):
				fst = self.states.index(rec[i])
				tst = self.states.index(rec[i+1])
				odds = self.stTranPr[pc][fst][tst] / self.stTranPr[nc][fst][tst]
				lodds += math.log(odds)
			prc = pc if lodds > thold else nc
			ypr.append(prc)
			if validate:
				yac.append(rec[klen])
			else:
				recp = prc + "\t" + ",".join(rec)
				print(recp)

		re = (yac, ypr)
		return re
	
	def __toIntClabel(self, labels):
		"""
		convert string class label to int
		
		Parameters
			labels : class label values
		"""
		return list(map(lambda l : self.clabels.index(l), labels))
	
		
