#!/Users/pranab/Tools/anaconda/bin/python

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
import numpy as np
import sklearn as sk
import random
import jprops
from util import *

#configuration management
class Configuration:
	def __init__(self, configFile, defValues, verbose=False):
		configs = {}
		with open(configFile) as fp:
  			for key, value in jprops.iter_properties(fp):
				configs[key] = value
		self.configs = configs
		self.defValues = defValues
		self.verbose = verbose

	#set config param
	def setParam(self, name, value):
		self.configs[name] = value

	# get string param
	def getStringConfig(self, name):
		if self.isNone(name):
			val = (None, False)
		elif self.isDefault(name):
			val = (self.handleDefault(name), True)
		else:
			val = (self.configs[name], False)
		if self.verbose:
			print "%s %s %s" %(name, self.configs[name], val[0])
		return val

	# get int param
	def getIntConfig(self, name):
		#print "%s %s" %(name,self.configs[name])
		if self.isNone(name):
			val = (None, False)
		elif self.isDefault(name):
			val = (self.handleDefault(name), True)
		else:
			val = (int(self.configs[name]), False)
		if self.verbose:
			print "%s %s %d" %(name, self.configs[name], val[0])
		return val
		
	# get float param
	def getFloatConfig(self, name):
		#print "%s %s" %(name,self.configs[name])
		if self.isNone(name):
			val = (None, False)
		elif self.isDefault(name):
			val = (self.handleDefault(name), True)
		else:
			val = (float(self.configs[name]), False)
		if self.verbose:
			print "%s %s %.3f" %(name, self.configs[name], val[0])
		return val

	# get boolean param
	def getBooleanConfig(self, name):
		if self.isNone(name):
			val = (None, False)
		elif self.isDefault(name):
			val = (self.handleDefault(name), True)
		else:
			bVal = self.configs[name].lower() == "true"
			val = (bVal, False)
		if self.verbose:
			print "%s %s %s" %(name, self.configs[name], val[0])
		return val
		
	# get int list param
	def getIntListConfig(self, name, delim):
		delSepStr = self.getStringConfig(name)
		intList = strToIntArray(delSepStr[0], delim)
		return (intList, delSepStr[1])

	def handleDefault(self, name):
		dVal = self.defValues[name]
		if (dVal[1] is None):
			val = dVal[0]
		else:
			raise ValueError(dVal[1])
		return val
		
	def isNone(self, name):
		return self.configs[name].lower() == "none"
		
	def isDefault(self, name):
		de = self.configs[name] == "_"
		#print de
		return de
		
	def eitherOrStringConfig(self, firstName, secondName):
		if not self.isNone(firstName):
			first = self.getStringConfig(firstName)[0]
			second = None
			if not self.isNone(secondName):
				raise ValueError("only one of the two parameters should be set and not both " + firstName + "  " + secondName)
		else:
			if not self.isNone(secondName):
				second = self.getStringConfig(secondtName)[0]
				first = None
			else:
				raise ValueError("at least one of the two parameters should be set " + firstName + "  " + secondName)
		return (first, second)

	def eitherOrIntConfig(self, firstName, secondName):
		if not self.isNone(firstName):
			first = self.getIntConfig(firstName)[0]
			second = None
			if not self.isNone(secondName):
				raise ValueError("only one of the two parameters should be set and not both " + firstName + "  " + secondName)
		else:
			if not self.isNone(secondName):
				second = self.getIntConfig(secondsName)[0]
				first = None
			else:
				raise ValueError("at least one of the two parameters should be set " + firstName + "  " + secondName)
		return (first, second)
	
#loads delim separated file and extracts columns
def loadDataFile(file, delim, cols, colIndices):
	data = np.loadtxt(file, delimiter=delim, usecols=cols)
	extrData = data[:,colIndices]
	return (data, extrData)

#extracts columns
def extrColumns(arr, columns):
	return arr[:, columns]

# subsample feature and class label data	
def subSample(featData, clsData, subSampleRate, withReplacement):
	sampSize = int(featData.shape[0] * subSampleRate)
	sampledIndx = np.random.choice(featData.shape[0],sampSize, replace=withReplacement)
	sampFeat = featData[sampledIndx]
	sampCls = clsData[sampledIndx]
	return(sampFeat, sampCls)

