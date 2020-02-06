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
import numpy as np
import sklearn as sk
from sklearn import preprocessing
import random
from math import *
from decimal import Decimal
import jprops
from util import *
from sampler import *

class Configuration:
	"""
	Configuration management. Supports default value, mandatory value and typed value.
	"""
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
			print( "{} {} {}".format(name, self.configs[name], val[0]))
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
			print( "{} {} {}".format(name, self.configs[name], val[0]))
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
			print( "{} {} {:06.3f}".format(name, self.configs[name], val[0]))
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
			print( "{} {} {}".format(name, self.configs[name], val[0]))
		return val
		
	# get int list param
	def getIntListConfig(self, name, delim):
		delSepStr = self.getStringConfig(name)
		intList = strToIntArray(delSepStr[0], delim)
		return (intList, delSepStr[1])
	
	# handles default
	def handleDefault(self, name):
		dVal = self.defValues[name]
		if (dVal[1] is None):
			val = dVal[0]
		else:
			raise ValueError(dVal[1])
		return val
	
	# true is value is None	
	def isNone(self, name):
		return self.configs[name].lower() == "none"
	
	# true if the value is default	
	def isDefault(self, name):
		de = self.configs[name] == "_"
		#print de
		return de
	
	# returns one of two string parameters	
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

	# returns one of two int parameters	
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
	

class CatLabelGenerator:
	"""
	label generator for categorical variables
	"""
	def __init__(self,  catValues, delim):
		self.encoders = {}
		self.catValues = catValues
		self.delim = delim
		for k in self.catValues.keys():	
			le = sk.preprocessing.LabelEncoder()	
			le.fit(self.catValues[k])
			self.encoders[k] = le

	# encode row
	def processRow(self, row):	
		#print row
		rowArr = row.split(self.delim)
		for i in range(len(rowArr)):
			if (i in self.catValues):
				curVal = rowArr[i]
				assert curVal in self.catValues[i], "categorival value invalid"
				encVal = self.encoders[i].transform([curVal])
				rowArr[i] = str(encVal[0])
		return self.delim.join(rowArr)		

	# get original labels
	def getOrigLabels(self, indx):
		return self.encoders[indx].classes_	


def loadDataFile(file, delim, cols, colIndices):
	"""
	loads delim separated file and extracts columns
	"""
	data = np.loadtxt(file, delimiter=delim, usecols=cols)
	extrData = data[:,colIndices]
	return (data, extrData)

def loadFeatDataFile(file, delim, cols):
	"""
	loads delim separated file and extracts columns
	"""
	data = np.loadtxt(file, delimiter=delim, usecols=cols)
	return data

def extrColumns(arr, columns):
	"""
	extracts columns
	"""
	return arr[:, columns]

def subSample(featData, clsData, subSampleRate, withReplacement):
	"""
	subsample feature and class label data	
	"""
	sampSize = int(featData.shape[0] * subSampleRate)
	sampledIndx = np.random.choice(featData.shape[0],sampSize, replace=withReplacement)
	sampFeat = featData[sampledIndx]
	sampCls = clsData[sampledIndx]
	return(sampFeat, sampCls)

def euclideanDistance(x,y):
	"""
	euclidean distance
	"""
	return sqrt(sum(pow(a-b, 2) for a, b in zip(x, y)))

def squareRooted(x):
	"""
	square root of sum square
	"""
	return round(sqrt(sum([a*a for a in x])),3)

def cosineSimilarity(x,y):
	"""
	cosine similarity
	"""
	numerator = sum(a*b for a,b in zip(x,y))
	denominator = squareRooted(x) * squareRooted(y)
	return round(numerator / float(denominator), 3)

def cosineDistance(x,y):
	"""
	cosine distance
	"""
	return 1.0 - cosineSimilarity(x,y)

def manhattanDistance(x,y):
	"""
	manhattan distance
	"""
	return sum(abs(a-b) for a,b in zip(x,y))

def nthRoot(value, nRoot):
	"""
	nth root
	"""
	rootValue = 1/float(nRoot)
	return round (Decimal(value) ** Decimal(rootValue),3)

def minkowskiDistance(x,y,pValue):
	"""
	minkowski distance
	"""
	return nthRoot(sum(pow(abs(a-b),pValue) for a,b in zip(x, y)), pValue)

def jaccardSimilarityX(x,y):
	"""
	jaccard similarity
	"""
	intersectionCardinality = len(set.intersection(*[set(x), set(y)]))
	unionCardinality = len(set.union(*[set(x), set(y)]))
	return intersectionCardinality/float(unionCardinality)

def jaccardSimilarity(x,y,wx=1.0,wy=1.0):
	"""
	jaccard similarity
	"""
	sx = set(x)
	sy = set(y)
	sxyInt = sx.intersection(sy)
	intCardinality = len(sxyInt)
	sxIntDiff = sx.difference(sxyInt)
	syIntDiff = sy.difference(sxyInt)
	unionCardinality = len(sx.union(sy))
	return intCardinality/float(intCardinality + wx * len(sxIntDiff) + wy * len(syIntDiff))

def norm(values, po=2):
	"""
	norm
	"""
	no = sum(list(map(lambda v: pow(v,po), values)))
	no = pow(no,1.0/po)
	return list(map(lambda v: v/no, values))
	
def createOneHotVec(size, indx = -1):
	"""
	random one hot vector
	"""
	vec = [0] * size
	s = random.randint(0, size - 1) if indx < 0 else indx
	vec[s] = 1
	return vec

def createAllOneHotVec(size):
	"""
	create all one hot vectors
	"""
	vecs = list()
	for i in range(size):
		vec = [0] * size
		vec[i] = 1
		vecs.append(vec)
	return vecs

def blockShuffle(data, blockSize):
	"""
	block shuffle 	
	"""
	numBlock = int(len(data) / blockSize)
	remain = len(data) % blockSize
	numBlock +=  (1 if remain > 0 else 0)
	shuffled = list()
	for i in range(numBlock):
		b = random.randint(0, numBlock-1)
		beg = b * blockSize
		if (b < numBlock-1):
			end = beg + blockSize
			shuffled.extend(data[beg:end])		
		else:
			shuffled.extend(data[beg:])
	return shuffled	

def shuffle(data, numShuffle):
	"""
	shuffle data
	"""
	sz = len(data)
	if numShuffle is None:
		numShuffle = int(sz / 2)
	for i in range(numShuffle):
		fi = random.randint(0, sz -1)
		se = random.randint(0, sz -1)
		tmp = data[fi]
		data[fi] = data[se]
		data[se] = tmp	

def randomWalk(size, start, lowStep, highStep):
	"""
	random walk	
	"""
	cur = start
	for i in range(size):
		yield cur
		cur += randomFloat(lowStep, highStep)

def binaryEcodeCategorical(values, value):
	"""
	one hot binary encoding	
	"""
	size = len(values)
	vec = [0] * size
	for i in range(size):
		if (values[i] == value):
			vec[i] = 1
	return vec		

def createLabeledSeq(inputData, tw):
	"""
	Creates feature, label pair from sequence data	
	"""
	features = list()
	labels = list()
	l = len(inputDta)
	for i in range(l - tw):
		trainSeq = inputData[i:i+tw]
		trainLabel = inputData[i+tw]
		features.append(trainSeq)
		labels.append(trainLabel)
	return (features, labels)

def createLabeledSeq(filePath, delim, index, tw):
	"""
	Creates feature, label pair from sequence data in file	
	"""
	seqData = getFileColumnAsFloat(filePath, delim, index)
	return createLabeledSeq(seqData, tw)
	
def difference(data, interval=1):
	"""
	takes difference in time series data
	"""
	diff = list()
	for i in range(interval, len(data)):
		value = data[i] - data[i - interval]
		diff.append(value)
	return diff
	
def normalizeMatrix(data, norm, axis=1):
	"""
	normalized each row of the matrix
	"""
	normalized = preprocessing.normalize(data,norm=norm, axis=axis)
	return normalized
	
def standardizeMatrix(data, axis=0):
	"""
	standardizes each column of the matrix with mean and std deviation
	"""
	standardized = preprocessing.scale(data, axis=axis)
	return standardized

def asNumpyArray(data):
	"""
	converts to numpy array
	"""
	return np.array(data)
