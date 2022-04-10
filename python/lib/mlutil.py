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
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
import random
from math import *
from decimal import Decimal
import statistics
import jprops
from Levenshtein import distance as ld
from util import *
from sampler import *

class Configuration:
	"""
	Configuration management. Supports default value, mandatory value and typed value.
	"""
	def __init__(self, configFile, defValues, verbose=False):
		"""
		initializer
		
		Parameters
			configFile : config file path
			defValues : dictionary of default values
			verbose : verbosity flag
		"""
		configs = {}
		with open(configFile) as fp:
  			for key, value in jprops.iter_properties(fp):
  				configs[key] = value
		self.configs = configs
		self.defValues = defValues
		self.verbose = verbose

	def override(self, configFile):
		"""
		over ride configuration from file
		
		Parameters
			configFile : override config file path
		"""
		with open(configFile) as fp:
  			for key, value in jprops.iter_properties(fp):
  				self.configs[key] = value
  			
	
	def setParam(self, name, value):
		"""
		override individual configuration

		Parameters
			name : config param name
			value : config param value
		"""
		self.configs[name] = value

	
	def getStringConfig(self, name):
		"""
		get string param

		Parameters
			name : config param name
		"""
		if self.isNone(name):
			val = (None, False)
		elif self.isDefault(name):
			val = (self.handleDefault(name), True)
		else:
			val = (self.configs[name], False)
		if self.verbose:
			print( "{} {} {}".format(name, self.configs[name], val[0]))
		return val

	
	def getIntConfig(self, name):
		"""
		get int param

		Parameters
			name : config param name
		"""
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
		
	
	def getFloatConfig(self, name):
		"""
		get float param

		Parameters
			name : config param name
		"""
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

	
	def getBooleanConfig(self, name):
		"""
		#get boolean param

		Parameters
			name : config param name
		"""
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
		
	
	def getIntListConfig(self, name, delim=","):
		"""
		get int list param

		Parameters
			name : config param name
			delim : delemeter
		"""
		if self.isNone(name):
			val = (None, False)
		elif self.isDefault(name):
			val = (self.handleDefault(name), True)
		else:
			delSepStr = self.getStringConfig(name)
		
			#specified as range
			intList = strListOrRangeToIntArray(delSepStr[0])
			val =(intList, delSepStr[1])
		return val
	
	def getFloatListConfig(self, name, delim=","):
		"""
		get float list param

		Parameters
			name : config param name
			delim : delemeter
		"""
		delSepStr = self.getStringConfig(name)
		if self.isNone(name):
			val = (None, False)
		elif self.isDefault(name):
			val = (self.handleDefault(name), True)
		else:
			flList = strToFloatArray(delSepStr[0], delim)
			val =(flList, delSepStr[1])
		return val

	
	def getStringListConfig(self, name, delim=","):
		"""
		get string list param

		Parameters
			name : config param name
			delim : delemeter
		"""
		delSepStr = self.getStringConfig(name)
		if self.isNone(name):
			val = (None, False)
		elif self.isDefault(name):
			val = (self.handleDefault(name), True)
		else:
			strList = delSepStr[0].split(delim)
			val = (strList, delSepStr[1])
		return val
	
	def handleDefault(self, name):
		"""
		handles default

		Parameters
			name : config param name
		"""
		dVal = self.defValues[name]
		if (dVal[1] is None):
			val = dVal[0]
		else:
			raise ValueError(dVal[1])
		return val
	
	
	def isNone(self, name):
		"""
		true is value is None	

		Parameters
			name : config param name
		"""
		return self.configs[name].lower() == "none"
	
	
	def isDefault(self, name):
		"""
		true if the value is default	

		Parameters
			name : config param name
		"""
		de = self.configs[name] == "_"
		#print de
		return de
	
	
	def eitherOrStringConfig(self, firstName, secondName):
		"""
		returns one of two string parameters	

		Parameters
			firstName : first parameter name
			secondName : second parameter name	
		"""
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
		"""
		returns one of two int parameters	

		Parameters
			firstName : first parameter name
			secondName : second parameter name	
		"""
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
		"""
		initilizers
		
		Parameters
			catValues : dictionary of categorical values
			delim : delemeter
		"""
		self.encoders = {}
		self.catValues = catValues
		self.delim = delim
		for k in self.catValues.keys():	
			le = preprocessing.LabelEncoder()	
			le.fit(self.catValues[k])
			self.encoders[k] = le

	def processRow(self, row):	
		"""
		encode row categorical values
		
		Parameters:
			row : data row
		"""
		#print row
		rowArr = row.split(self.delim)
		for i in range(len(rowArr)):
			if (i in self.catValues):
				curVal = rowArr[i]
				assert curVal in self.catValues[i], "categorival value invalid"
				encVal = self.encoders[i].transform([curVal])
				rowArr[i] = str(encVal[0])
		return self.delim.join(rowArr)		

	def getOrigLabels(self, indx):
		"""
		get original labels
		
		Parameters:
			indx : column index
		"""
		return self.encoders[indx].classes_	


class SupvLearningDataGenerator:
	"""
	data generator for supervised learning
	"""
	def __init__(self,  configFile):
		"""
		initilizers
		
		Parameters
			configFile : config file path
		"""
		defValues = dict()
		defValues["common.num.samp"] = (100, None)
		defValues["common.num.feat"] = (5, None)
		defValues["common.feat.trans"] = (None, None)
		defValues["common.feat.types"] = (None, "missing feature types")
		defValues["common.cat.feat.distr"] = (None, None)
		defValues["common.output.precision"] = (3, None)
		defValues["common.error"] = (0.01, None)
		defValues["class.gen.technique"] = ("blob", None)
		defValues["class.num.feat.informative"] = (2, None)
		defValues["class.num.feat.redundant"] = (2, None)
		defValues["class.num.feat.repeated"] = (0, None)
		defValues["class.num.feat.cat"] = (0, None)
		defValues["class.num.class"] = (2, None)

		self.config = Configuration(configFile, defValues)

	def genClassifierData(self):
		"""
		generates classifier data
		"""
		nsamp =  self.config.getIntConfig("common.num.samp")[0]
		nfeat =  self.config.getIntConfig("common.num.feat")[0]
		nclass =  self.config.getIntConfig("class.num.class")[0]
		#transform with shift and scale
		ftrans =  self.config.getFloatListConfig("common.feat.trans")[0]
		feTrans = dict()
		for i in range(0, len(ftrans), 2):
			tr = (ftrans[i], ftrans[i+1])
			indx = int(i/2)
			feTrans[indx] = tr

		ftypes =  self.config.getStringListConfig("common.feat.types")[0]

		# categorical feature distribution
		feCatDist = dict()
		fcatdl =  self.config.getStringListConfig("common.cat.feat.distr")[0]
		for fcatds in fcatdl:
			fcatd = fcatds.split(":")
			feInd =  int(fcatd[0])
			clVal =  int(fcatd[1])
			key = (feInd, clVal)		#feature index and class value
			dist = list(map(lambda i : (fcatd[i], float(fcatd[i+1])), range(2, len(fcatd), 2)))
			feCatDist[key] = CategoricalRejectSampler(*dist)

		#shift and scale
		genTechnique = self.config.getStringConfig("class.gen.technique")[0]
		error = self.config.getFloatConfig("common.error")[0]
		if genTechnique == "blob":
			features, claz = make_blobs(n_samples=nsamp, centers=nclass, n_features=nfeat)
			for i in range(nsamp):			#shift and scale
				for j in range(nfeat):
					tr = feTrans[j]
					features[i,j] = (features[i,j]  + tr[0]) * tr[1]
			claz = np.array(list(map(lambda c : random.randint(0, nclass-1) if random.random() < error else c, claz)))
		elif genTechnique == "classify":
			nfeatInfo =  self.config.getIntConfig("class.num.feat.informative")[0]
			nfeatRed =  self.config.getIntConfig("class.num.feat.redundant")[0]
			nfeatRep =  self.config.getIntConfig("class.num.feat.repeated")[0]
			shifts = list(map(lambda i : feTrans[i][0], range(nfeat)))
			scales = list(map(lambda i : feTrans[i][1], range(nfeat)))
			features, claz = make_classification(n_samples=nsamp, n_features=nfeat, n_informative=nfeatInfo, n_redundant=nfeatRed, 
			n_repeated=nfeatRep, n_classes=nclass, flip_y=error, shift=shifts, scale=scales)
		else:
			raise "invalid genaration technique"

		# add categorical features and format
		nCatFeat = self.config.getIntConfig("class.num.feat.cat")[0]
		prec =  self.config.getIntConfig("common.output.precision")[0]
		for f , c in zip(features, claz):
			nfs = list(map(lambda i : self.numFeToStr(i, f[i], c, ftypes[i], prec), range(nfeat)))
			if nCatFeat > 0:
				cfs = list(map(lambda i : self.catFe(i, c, ftypes[i], feCatDist), range(nfeat, nfeat + nCatFeat, 1)))
				rec = ",".join(nfs) + "," +  ",".join(cfs)  + "," + str(c)
			else:
				rec = ",".join(nfs)  + "," + str(c)
			yield rec

	def numFeToStr(self, fv, ft, prec):
		"""
		nummeric feature value to string
		
		Parameters
			fv : field value
			ft : field data type
			prec : precision
		"""
		if ft == "float":
			s = formatFloat(prec, fv)
		elif ft =="int":
			s = str(int(fv))
		else:		
			raise "invalid type expecting float or int"
		return s

	def catFe(self, i, cv, ft, feCatDist):
		"""
		generate categorical feature
		
		Parameters
			i : col index
			cv : class value
			ft : field data type
			feCatDist : cat value distribution
		"""
		if ft == "cat":
			key = (i, cv)
			s = feCatDist[key].sample()
		else:		
			raise "invalid type expecting categorical"
		return s



def loadDataFile(file, delim, cols, colIndices):
	"""
	loads delim separated file and extracts columns

	Parameters
		file : file path
		delim : delemeter
		cols : columns to use from file
		colIndices ; columns to extract
	"""
	data = np.loadtxt(file, delimiter=delim, usecols=cols)
	extrData = data[:,colIndices]
	return (data, extrData)

def loadFeatDataFile(file, delim, cols):
	"""
	loads delim separated file and extracts columns
	
	Parameters
		file : file path
		delim : delemeter
		cols : columns to use from file
	"""
	data = np.loadtxt(file, delimiter=delim, usecols=cols)
	return data

def extrColumns(arr, columns):
	"""
	extracts columns
	
	Parameters
		arr : 2D array
		columns : columns
	"""
	return arr[:, columns]

def subSample(featData, clsData, subSampleRate, withReplacement):
	"""
	subsample feature and class label data	

	Parameters
		featData : 2D array of feature data
		clsData : arrray of class labels
		subSampleRate : fraction to be sampled
		withReplacement : true if sampling with replacement
	"""
	sampSize = int(featData.shape[0] * subSampleRate)
	sampledIndx = np.random.choice(featData.shape[0],sampSize, replace=withReplacement)
	sampFeat = featData[sampledIndx]
	sampCls = clsData[sampledIndx]
	return(sampFeat, sampCls)

def euclideanDistance(x,y):
	"""
	euclidean distance

	Parameters
		x : first vector
		y : second fvector
	"""
	return sqrt(sum(pow(a-b, 2) for a, b in zip(x, y)))

def squareRooted(x):
	"""
	square root of sum square

	Parameters
		x : data vector
	"""
	return round(sqrt(sum([a*a for a in x])),3)

def cosineSimilarity(x,y):
	"""
	cosine similarity
	
	Parameters
		x : first vector
		y : second fvector
	"""
	numerator = sum(a*b for a,b in zip(x,y))
	denominator = squareRooted(x) * squareRooted(y)
	return round(numerator / float(denominator), 3)

def cosineDistance(x,y):
	"""
	cosine distance

	Parameters
		x : first vector
		y : second fvector
	"""
	return 1.0 - cosineSimilarity(x,y)

def manhattanDistance(x,y):
	"""
	manhattan distance

	Parameters
		x : first vector
		y : second fvector
	"""
	return sum(abs(a-b) for a,b in zip(x,y))

def nthRoot(value, nRoot):
	"""
	nth root

	Parameters
		value : data value
		nRoot : root
	"""
	rootValue = 1/float(nRoot)
	return round (Decimal(value) ** Decimal(rootValue),3)

def minkowskiDistance(x,y,pValue):
	"""
	minkowski distance

	Parameters
		x : first vector
		y : second fvector
		pValue : power factor
	"""
	return nthRoot(sum(pow(abs(a-b),pValue) for a,b in zip(x, y)), pValue)

def jaccardSimilarityX(x,y):
	"""
	jaccard similarity

	Parameters
		x : first vector
		y : second fvector
	"""
	intersectionCardinality = len(set.intersection(*[set(x), set(y)]))
	unionCardinality = len(set.union(*[set(x), set(y)]))
	return intersectionCardinality/float(unionCardinality)

def jaccardSimilarity(x,y,wx=1.0,wy=1.0):
	"""
	jaccard similarity
	
	Parameters
		x : first vector
		y : second fvector
		wx : weight for x
		wy : weight for y
	"""
	sx = set(x)
	sy = set(y)
	sxyInt = sx.intersection(sy)
	intCardinality = len(sxyInt)
	sxIntDiff = sx.difference(sxyInt)
	syIntDiff = sy.difference(sxyInt)
	unionCardinality = len(sx.union(sy))
	return intCardinality/float(intCardinality + wx * len(sxIntDiff) + wy * len(syIntDiff))

def levenshteinSimilarity(s1, s2):
	"""
	Levenshtein similarity for strings
	
	Parameters
		sx : first string
		sy : second string
	"""
	assert type(s1) == str and type(s2) == str,  "Levenshtein similarity is for string only"
	d = ld(s1,s2)
	#print(d)
	l = max(len(s1),len(s2))
	d = 1.0 - min(d/l, 1.0)
	return d	

def norm(values, po=2):
	"""
	norm

	Parameters
		values : list of values
		po : power
	"""
	no = sum(list(map(lambda v: pow(v,po), values)))
	no = pow(no,1.0/po)
	return list(map(lambda v: v/no, values))
	
def createOneHotVec(size, indx = -1):
	"""
	random one hot vector
	
	Parameters
		size : vector size
		indx : one hot position
	"""
	vec = [0] * size
	s = random.randint(0, size - 1) if indx < 0 else indx
	vec[s] = 1
	return vec

def createAllOneHotVec(size):
	"""
	create all one hot vectors
	
	Parameters
		size : vector size and no of vectors
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
	
	Parameters
		data : list data
		blockSize : block size
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
	shuffle data by randonm swapping
	
	Parameters
		data : list data
		numShuffle : no of pairwise swaps
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
	
	Parameters
		size : list data
		start : initial position
		lowStep : step min
		highStep : step max
	"""
	cur = start
	for i in range(size):
		yield cur
		cur += randomFloat(lowStep, highStep)

def binaryEcodeCategorical(values, value):
	"""
	one hot binary encoding	
	
	Parameters
		values : list of values
		value : value to be replaced with 1
	"""
	size = len(values)
	vec = [0] * size
	for i in range(size):
		if (values[i] == value):
			vec[i] = 1
	return vec		

def createLabeledSeq(inputData, tw):
	"""
	Creates feature, label pair from sequence data, where we have tw number of features followed by output
	
	Parameters
		values : list containing feature and label
		tw : no of features
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
	Creates feature, label pair from 1D sequence data in file	
	
	Parameters
		filePath : file path
		delim : delemeter
		index : column index
		tw : no of features
	"""
	seqData = getFileColumnAsFloat(filePath, delim, index)
	return createLabeledSeq(seqData, tw)

def fromMultDimSeqToTabular(data, inpSize, seqLen):
	"""
	Input shape (nrow, inpSize * seqLen) output shape(nrow * seqLen, inpSize)
	
	Parameters
		data : 2D array
		inpSize : each input size in sequence
		seqLen : sequence length
	"""	
	nrow = data.shape[0]
	assert data.shape[1] == inpSize * seqLen, "invalid input size or sequence length"
	return data.reshape(nrow * seqLen, inpSize)
	
def fromTabularToMultDimSeq(data, inpSize, seqLen):
	"""
	Input shape (nrow * seqLen, inpSize)   output  shape (nrow, inpSize * seqLen) 

	Parameters
		data : 2D array
		inpSize : each input size in sequence
		seqLen : sequence length
	"""	
	nrow = int(data.shape[0] / seqLen)
	assert data.shape[1] == inpSize, "invalid input size"
	return data.reshape(nrow,  seqLen * inpSize)

def difference(data, interval=1):
	"""
	takes difference in time series data

	Parameters
		data :list data
		interval : interval for difference
	"""
	diff = list()
	for i in range(interval, len(data)):
		value = data[i] - data[i - interval]
		diff.append(value)
	return diff
	
def normalizeMatrix(data, norm, axis=1):
	"""
	normalized each row of the matrix
	
	Parameters
		data : 2D data
		nporm : normalization method
		axis : row or column
	"""
	normalized = preprocessing.normalize(data,norm=norm, axis=axis)
	return normalized
	
def standardizeMatrix(data, axis=0):
	"""
	standardizes each column of the matrix with mean and std deviation

	Parameters
		data : 2D data
		axis : row or column
	"""
	standardized = preprocessing.scale(data, axis=axis)
	return standardized

def asNumpyArray(data):
	"""
	converts to numpy array

	Parameters
		data  : array
	"""
	return np.array(data)

def perfMetric(metric, yActual, yPred, clabels=None):
	"""
	predictive model accuracy metric

	Parameters
		metric : accuracy metric
		yActual : actual values array
		yPred : predicted values array
		clabels : class labels
	"""
	if metric == "rsquare":
		score = metrics.r2_score(yActual, yPred)
	elif metric == "mae":
		score = metrics.mean_absolute_error(yActual, yPred)
	elif metric == "mse":
		score = metrics.mean_squared_error(yActual, yPred)
	elif metric == "acc":
		yPred = np.rint(yPred)
		score = metrics.accuracy_score(yActual, yPred)
	elif metric == "mlAcc":
		yPred = np.argmax(yPred, axis=1)
		score = metrics.accuracy_score(yActual, yPred)
	elif metric == "prec":
		yPred = np.argmax(yPred, axis=1)
		score = metrics.precision_score(yActual, yPred)
	elif metric == "rec":
		yPred = np.argmax(yPred, axis=1)
		score = metrics.recall_score(yActual, yPred)
	elif metric == "fone":
		yPred = np.argmax(yPred, axis=1)
		score = metrics.f1_score(yActual, yPred)
	elif metric == "confm":
		yPred = np.argmax(yPred, axis=1)
		score = metrics.confusion_matrix(yActual, yPred)
	elif metric == "clarep":
		yPred = np.argmax(yPred, axis=1)
		score = metrics.classification_report(yActual, yPred)
	elif metric == "bce":
		if clabels is None:
			clabels = [0, 1]
		score = metrics.log_loss(yActual, yPred, labels=clabels)
	elif metric == "ce":
		assert clabels is not None, "labels must be provided"
		score = metrics.log_loss(yActual, yPred, labels=clabels)
	else:
		exitWithMsg("invalid prediction performance metric " + metric)
	return score

def scaleData(data, method):
	"""
	scales feature data column wise

	Parameters
		data : 2D array
		method : scaling method
	"""
	if method == "minmax":
		scaler = preprocessing.MinMaxScaler()
		data = scaler.fit_transform(data)
	elif method == "zscale":
		data = preprocessing.scale(data)	
	else:
		raise ValueError("invalid scaling method")	
	return data

def scaleDataWithParams(data, method, scParams):
	"""
	scales feature data column wise

	Parameters
		data : 2D array
		method : scaling method
		scParams : scaling parameters
	"""
	if method == "minmax":
		data = scaleMinMaxTabData(data, scParams)
	elif method == "zscale":
		raise ValueError("invalid scaling method")	
	else:
		raise ValueError("invalid scaling method")	
	return data


def scaleMinMaxTabData(tdata, minMax):
	"""
	for tabular scales feature data column wise using min max values for each field

	Parameters
		tdata : 2D array
		minMax : ni, max and range for each column
	"""
	stdata = list()
	for r in tdata:
		srdata = list()
		for i, c in enumerate(r):
			sd = (c - minMax[i][0]) / minMax[i][2]
			srdata.append(sd)
		stdata.append(srdata)
	return stdata
	
def scaleMinMax(rdata, minMax):
	"""
	scales feature data column wise using min max values for each field

	Parameters
		rdata : data array
		minMax : ni, max and range for each column
	"""
	srdata = list()
	for i in range(len(rdata)):
		d = rdata[i]
		sd = (d - minMax[i][0]) / minMax[i][2]
		srdata.append(sd)
	return srdata
	
def harmonicNum(n):
	"""
	harmonic number

	Parameters
		n : number
	"""
	h = 0
	for i in range(1, n+1, 1):
		h += 1.0 / i
	return h
	
def digammaFun(n):
	"""
	figamma function

	Parameters
		n : number
	"""
	#Euler Mascheroni constant
	ec = 0.577216
	return harmonicNum(n - 1) - ec
			
def getDataPartitions(tdata, types, columns = None):
	"""
	partitions data with the given columns and random split point defined with predicates

	Parameters
		tdata : 2D array
		types : data typers
		columns : column indexes
	"""
	(dtypes, cvalues) = extractTypesFromString(types)
	if columns is None:
		ncol = len(data[0])
		columns = list(range(ncol))
	ncol = len(columns)
	#print(columns)
		
	# partition predicates
	partitions = None
	for c in columns:
		#print(c)
		dtype = dtypes[c]
		pred = list()
		if dtype == "int" or dtype == "float":
			(vmin, vmax) = getColMinMax(tdata, c)
			r = vmax - vmin
			rmin = vmin + .2 * r
			rmax = vmax - .2 * r
			sp = randomFloat(rmin, rmax)
			if dtype == "int":
				sp = int(sp)
			else:
				sp = "{:.3f}".format(sp)
				sp = float(sp)
			pred.append([c, "LT", sp])
			pred.append([c, "GE", sp])
		elif dtype == "cat":
			cv = cvalues[c]
			card = len(cv) 
			if card < 3:
				num = 1
			else:
				num = randomInt(1, card - 1)
			sp = selectRandomSubListFromList(cv, num)
			sp = " ".join(sp)
			pred.append([c, "IN", sp])
			pred.append([c, "NOTIN", sp])
		
		#print(pred)
		if partitions is None:
			partitions = pred.copy()
			#print("initial")
			#print(partitions)
		else:
			#print("extension")
			tparts = list()
			for p in partitions:
				#print(p)
				l1 = p.copy()
				l1.extend(pred[0])
				l2 = p.copy()
				l2.extend(pred[1])
				#print("after extension")
				#print(l1)
				#print(l2)
				tparts.append(l1)
				tparts.append(l2)
			partitions = tparts	
			#print("extending")
			#print(partitions)
	
	#for p in partitions:
		#print(p)	
	return partitions			
		
def genAlmostUniformDistr(size, nswap=50):
	"""
	generate probability distribution
	
	Parameters
		size : distr size
		nswap : no of mass swaps
	"""
	un = 1.0 / size
	distr = [un] * size
	distr = mutDistr(distr, 0.1 * un, nswap)
	return distr

def mutDistr(distr, shift, nswap=50):
	"""
	mutates a probability distribution
	
	Parameters
		distr distribution
		shift : amount of shift for swap
		nswap : no of mass swaps
	"""
	size = len(distr)
	for _ in range(nswap):
		fi = randomInt(0, size -1)
		si = randomInt(0, size -1)
		while fi == si:
			fi = randomInt(0, size -1)
			si = randomInt(0, size -1)
		
		shift = randomFloat(0, shift)
		t = distr[fi]
		distr[fi] -= shift
		if (distr[fi] < 0):
			distr[fi] = 0.0
			shift = t
		distr[si] += shift
	return distr

def generateBinDistribution(size, ntrue):
	"""
	generate binary array with some elements set to 1
	
	Parameters
		size : distr size
		ntrue : no of true values
	"""
	distr = [0] * size
	idxs = selectRandomSubListFromList(list(range(size)), ntrue)
	for i in idxs:
		distr[i] = 1
	return distr

def mutBinaryDistr(distr, nmut):
	"""
	mutate binary distribution
	
	Parameters
		distr : distr
		nmut : no of mutations
	"""
	idxs = selectRandomSubListFromList(list(range(len(distr))), nmut)
	for i in idxs:
		distr[i] = distr[i] ^ 1
	
	
def fileSelFieldSubSeqModifierGen(filePath, column, offset, seqLen, modifier, precision, delim=","):
	"""
	file record generator that superimposes given data in the specified segment of a column

	Parameters
		filePath ; file path
		column : column index 
		offset : offset into column values
		seqLen : length of subseq
		modifier : data to be superimposed either list or a sampler object
		precision : floating point precision
		delim : delemeter
	"""
	beg = offset
	end = beg + seqLen
	isList = type(modifier) == list
	i = 0
	for rec in fileRecGen(filePath, delim):
		if i >= beg and i < end:
			va = float(rec[column])
			if isList:
				va += modifier[i - beg] 
			else:
				va += modifier.sample()
			rec[column] = formatFloat(precision, va)
		yield delim.join(rec)
		i += 1
	
class ShiftedDataGenerator:
	"""
	transforms data for distribution shift
	"""
	def __init__(self, types, tdata, addFact, multFact):
		"""
		initializer
		
		Parameters
			types data types
			tdata : 2D array
			addFact ; factor for data shift
			multFact ; factor for data scaling
		"""
		(self.dtypes, self.cvalues) = extractTypesFromString(types)
		
		self.limits = dict()
		for k,v in self.dtypes.items():
			if v == "int" or v == "false":
				(vmax, vmin) = getColMinMax(tdata, k)
				self.limits[k] = vmax - vmin
		self.addMin = - addFact / 2
		self.addMax =  addFact / 2
		self.multMin = 1.0 - multFact / 2
		self.multMax = 1.0 + multFact / 2
		
		
		
	
	def transform(self, tdata):
		"""
		linear transforms data to create  distribution shift with random shift and scale

		Parameters
			types : data types
		"""
		transforms = dict()
		for k,v in self.dtypes.items():
			if v == "int" or v == "false":				
				shift = randomFloat(self.addMin, self.addMax) * self.limits[k] 
				scale = randomFloat(self.multMin, self.multMax)
				trns = (shift, scale)
				transforms[k] = trns
			elif v == "cat":
				transforms[k] = isEventSampled(50)
				
		ttdata = list()
		for rec in tdata:
			nrec = rec.copy()
			for c in range(len(rec)):
				if c in self.dtypes:
					dtype = self.dtypes[c]
					if dtype == "int" or dtype == "float":
						(shift, scale) = transforms[c]
						nval = shift +  rec[c] * scale
						if dtype == "int":
							nrec[c] = int(nval)
						else:
							nrec[c] = nval
					elif dtype == "cat":
						cv = self.cvalues[c]
						if transforms[c]:
							nval = selectOtherRandomFromList(cv, rec[c])
							nrec[c] = nval
					
			ttdata.append(nrec)
			
		return ttdata
		
	def transformSpecified(self, tdata, sshift, scale):
		"""
		linear transforms data to create  distribution shift shift specified shift and scale

		Parameters
			types : data types
			sshift : shift factor
			scale : scale factor
		"""
		transforms = dict()
		for k,v in self.dtypes.items():
			if v == "int" or v == "false":				
				shift = sshift * self.limits[k] 
				trns = (shift, scale)
				transforms[k] = trns
			elif v == "cat":
				transforms[k] = isEventSampled(50)
				
		ttdata = self.__scaleShift(tdata, transforms)
		return ttdata
		
	def __scaleShift(self, tdata, transforms):
		"""
		shifts and scales tabular data
		
		Parameters
			tdata : 2D array
			transforms : transforms to apply
		"""
		ttdata = list()
		for rec in tdata:
			nrec = rec.copy()
			for c in range(len(rec)):
				if c in self.dtypes:
					dtype = self.dtypes[c]
					if dtype == "int" or dtype == "float":
						(shift, scale) = transforms[c]
						nval = shift + rec[c] * scale
						if dtype == "int":
							nrec[c] = int(nval)
						else:
							nrec[c] = nval
					elif dtype == "cat":
						cv = self.cvalues[c]
						if transforms[c]:
							#nval = selectOtherRandomFromList(cv, rec[c])
							#nrec[c] = nval
							pass
					
			ttdata.append(nrec)
		return ttdata
		
class RollingStat(object):
	"""
	stats for rolling windowt
	"""
	def __init__(self, wsize):
		"""
		initializer
		
		Parameters
			wsize : window size
		"""
		self.window = list()
		self.wsize = wsize
		self.mean = None
		self.sd = None

	def add(self, value):
		"""
		add a value
		
		Parameters
			value : value to add
		"""
		self.window.append(value)
		if len(self.window) > self.wsize:
			self.window = self.window[1:]
		
	def getStat(self):
		"""
		get rolling window mean and std deviation
		"""
		assertGreater(len(self.window), 0, "window is empty")
		if len(self.window) == 1:
			self.mean = self.window[0]
			self.sd = 0
		else:
			self.mean = statistics.mean(self.window)
			self.sd = statistics.stdev(self.window, xbar=self.mean)
		re = (self.mean, self.sd)
		return re
		
	def getSize(self):
		"""
		return window size
		"""
		return len(self.window)
		