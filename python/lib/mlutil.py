#!/Users/pranab/Tools/anaconda/bin/python

# Package imports
import os
import sys
import numpy as np
import sklearn as sk
import random
import jprops

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
