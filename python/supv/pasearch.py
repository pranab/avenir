#!/Users/pranab/Tools/anaconda/bin/python

# Package imports
import os
import sys
import numpy as np
import sklearn as sk
import random
import jprops
sys.path.append(os.path.abspath("../lib"))
from util import *

#enumerate through provided list of param values
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
			
			#number of values for each parameter
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
		
#random search through provided list of parameter values
class RandomParameterSearch:
	def __init__(self, verbose=False):
		self.verbose = verbose
		self.parameters = []
		self.paramData = {}	
		self.currentParams = []
		self.bestSolution = None
		self.curIter = 0

	# add param name and type
	def addParam(self, param):
		self.parameters.append(param)
	
	# add param data	
	def addParamVaues(self, paramName, paramData):
		self.paramData[paramName] = paramData
		
	# max iterations	
	def setMaxIter(self,maxIter):
		self.maxIter = maxIter

	# prepare
	def prepare(self):
		pass

	# next param combination
	def nextParamValues(self):
		retParamNameValue = None
		if (self.curIter < self.maxIter):
			retParamNameValue = []
			for pName, pValues in self.paramData.iteritems():
				pValue = selectRandomFromList(pValues)
				retParamNameValue.append((pName, pValue))
			self.curIter = self.curIter + 1
			self.currentParams = retParamNameValue
		return retParamNameValue
				
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
		