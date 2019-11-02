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
from random import randint
from datetime import datetime
from dateutil.parser import parse
import numpy as np
import sklearn as sk
from sklearn.externals import joblib
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../supv"))
from util import *
from mlutil import *
from rf import *
from svm import *
from gbt import *

# presicriptive analytic
class Prescriptor(object):
	def __init__(self, configFile):
		defValues = {}
		defValues["common.classifier"] = ("rf", None)
		defValues["common.classifier.config"] = (None, "missing classifier config file")
		defValues["common.class.var"] = ("0,1", None)
		defValues["common.cat.encoding"] = ("binary", None)
		defValues["common.feature.list"] = (None, "missing feture list")
		defValues["common.feature.num.grid"] = (20, None)
		defValues["common.feature.float.distr"] = (None, "missing float feature distribution")
		defValues["common.feature.int.range"] = (None, "missing int feature range")
		defValues["common.feature.cat.values"] = (None, "missing categorical feature values")

		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.initialize()
	
	# do all the initializations	
	def initialize(self):
		clfName =  self.config.getStringConfig("common.classifier")[0]
		clfConfigFile = self.config.getStringConfig("common.classifier.config")[0]
		
		if clfName == "rf":
			self.classifier = RandomForest(clfConfigFile)
		elif clfName == "gbt":
			self.classifier = GradientBoostedTrees(clfConfigFile)
		elif clfName == "svm":
			self.classifier = SupportVectorMachine(clfConfigFile)
		else:
			raise valueError("unsupported classifier")
		self.classVars = self.config.getStringConfig("common.class.var")[0].split(",")
		self.catEncoding = self.config.getStringConfig("common.cat.encoding")[0]
		
		# num of grids for float features
		self.numGrids = self.config.getIntConfig("common.feature.num.grid")[0]
		
		#feature type
		items= self.config.getStringConfig("common.feature.list")[0].split(",")
		self.featType = dict()
		for it in items:
			parts = it.split(":")
			self.featType[int(parts[0])] = parts[1]
		
		# float feature distr
		self.floatFeatDistr = {}	
		items = self.config.getStringConfig("common.feature.float.distr")[0].split(",")	
		for it in items:
			parts = it.split(":")
			self.floatFeatDistr[int(parts[0])] = float(parts[0])
		
		# int feature distr
		self.intFeatRange = {}	
		items = self.config.getStringConfig("common.feature.int.range")[0].split(",")	
		for it in items:
			parts = it.split(":")
			self.intFeatRange[int(parts[0])] = int(parts[0])
			
		# cat feature values
		self.catFeatValues = {}	
		items = self.config.getStringConfig("common.feature.cat.values")[0].split(",")	
		for it in items:
			parts = it.split(":")
			self.catFeatValues[int(parts[0])] = parts[1:]
		
	# model interpretation with independent conditional expectation	
	def indCondExp(self, rec, feature):
		fields = rec.split(",")
		fType = self.featType[feature]
		
		if fType == "int":
			rangeVal = self.intFeatRange[feature]
			refVal = int(fields[feature])
			lowVal = refVal - rangeVal
			step = 1
			numGrids = 2 * rangeVal
		elif fType == "float":
			rangeVal = self.floatFeatRange[feature]
			refVal = float(fields[feature])
			lowVal = refVal - rangeVal
			step = 2 * rangeVal / self.numGrids
			numGrids = self.numGrids
		elif fType == "categorical":
			catValues = self.catFeatValues[feature]
			lowVal = 0
			step = 1
			numGrids = len(catValues)
		else:
			raise ValueError("unsupported data type")
			
		results = list()
		
		# scan grid
		curVal = lowVal	
		for i in range(numGrids):
			if fType == "int":
				fields[feature] = str(curVal)
			elif fType == "float":
				curValStr = "%.3f" %(curVal)
				fields[feature] = curValStr
			elif fType == "categorical":
				cVal = catValues[i]
				vec = binaryEcodeCategorical(catValues, cVal)
				for j in range(numGrids):
					fields[feature + j] = str(vec[j])
				
				
			modRec = ",".join(fields)	
			clp = self.classifier.predictProb(modRec)
			result = (curVal, clp[0][1])
			results.append(result)
			curVal += step
			
		return results
		
		