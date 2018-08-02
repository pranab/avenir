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

#configuration management
class GuidedParameterSearch:
	def __init__(self, verbose=False):
		self.verbose = verbose
		self.parameters = []
		self.paramData = {}	
		self.paramIndexes = []
		self.numParamValues = []
		self.currentParams = []
		self.bestSolution = None
		
	# add param name and type
	def addParam(self, param):
		self.parameters.append(param)
	
	# add param data	
	def addParamVaues(self, paramName, paramData):
		self.paramData[paramName] = paramData
	
	# prepare
	def prepare(self):
		self.numParams = len(self.parameters)
		for i in range(self.numParams):
			self.paramIndexes.append(0)
			paramName = self.parameters[i][0]
			self.numParamValues.append(len(self.paramData[paramName]))
		self.curParamIndex = 0
		
		paramValueCombList = []
		paramValueComb = []
		paramValueCombList.append(paramValueComb)
		
		# all params
		for i in range(self.numParams):
			paramValueCombListTemp = []
			for paramValueComb in paramValueCombList:
				# all param values
				for j in range(self.numParamValues[i]):
					paramValueCombTemp = paramValueComb[:]
					paramValueCombTemp.append(j)
					paramValueCombListTemp.append(paramValueCombTemp)
			paramValueCombList = paramValueCombListTemp
		self.paramValueCombList = paramValueCombList
		self.numParamValueComb = len(self.paramValueCombList)
		self.curParamValueCombIndx = 0;
		
	# next param combination
	def nextParamValues(self):
		retParamNameValue = None
		if self.curParamValueCombIndx < len(self.paramValueCombList):
			retParamNameValue = []
			curParams = self.paramValueCombList[self.curParamValueCombIndx]
			print curParams
			for i in range(len(curParams)):
				paramName = self.parameters[i][0]
				paramValue = self.paramData[paramName][curParams[i]]
				retParamNameValue.append((paramName, paramValue))
			self.curParamValueCombIndx = self.curParamValueCombIndx + 1
			self.currentParams = retParamNameValue
		return 	retParamNameValue
			
	# set cost of current parameter set
	def setCost(self, cost):
		if self.bestSolution is not None:
			if cost < self.bestSolution[1]:
				self.bestSolution = (self.currentParams, cost)
		else:
			self.bestSolution = (self.currentParams, cost)
			
	# get best solution
	def getBestSolution(self):
		return 	self.bestSolution
