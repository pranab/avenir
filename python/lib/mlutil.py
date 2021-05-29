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
import sklearn as sk
from sklearn import preprocessing
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
import random
from math import *
from decimal import Decimal
import jprops
from Levenshtein import distance as ld
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

	def override(self, configFile):
		"""
		over ride configuration from file
		"""
		with open(configFile) as fp:
  			for key, value in jprops.iter_properties(fp):
  				self.configs[key] = value
  			
	
	def setParam(self, name, value):
		"""
		override individual configuration
		"""
		self.configs[name] = value

	
	def getStringConfig(self, name):
		"""
		get string param
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
		"""
		delSepStr = self.getStringConfig(name)
		intList = None
		
		#specified as range
		ra = delSepStr[0].split(":")
		if len(ra) > 1:
			intList = list(range(int(ra[0]), int(ra[1]) + 1, 1))
			
		if self.isNone(name):
			val = (None, False)
		elif self.isDefault(name):
			val = (self.handleDefault(name), True)
		else:
			if intList is None:
				intList = strToIntArray(delSepStr[0], delim)
			val =(intList, delSepStr[1])
		return val
	
	def getFloatListConfig(self, name, delim=","):
		"""
		get float list param
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
		"""
		return self.configs[name].lower() == "none"
	
	
	def isDefault(self, name):
		"""
		true if the value is default	
		"""
		de = self.configs[name] == "_"
		#print de
		return de
	
	
	def eitherOrStringConfig(self, firstName, secondName):
		"""
		returns one of two string parameters	
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


class SupvLearningDataGenerator:
	"""
	data generator for supervised learning
	"""
	def __init__(self,  configFile):
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

	def numFeToStr(self, i, fv, cv, ft, prec):
		"""
		nummeric feature value to string
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

def levenshteinSimilarity(s1, s2):
	"""
	Levenshtein similarity for strings
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
	Creates feature, label pair from sequence data, where we have tw number of features followed by output
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
	"""
	seqData = getFileColumnAsFloat(filePath, delim, index)
	return createLabeledSeq(seqData, tw)

def fromMultDimSeqToTabular(data, inpSize, seqLen):
	"""
	Input shape (nrow, inpSize * seqLen) output shape(nrow * seqLen, inpSize)
	"""	
	nrow = data.shape[0]
	assert data.shape[1] == inpSize * seqLen, "invalid input size or sequence length"
	return data.reshape(nrow * seqLen, inpSize)
	
def fromTabularToMultDimSeq(data, inpSize, seqLen):
	"""
	Input shape (nrow * seqLen, inpSize)   output  shape (nrow, inpSize * seqLen) 
	"""	
	nrow = int(data.shape[0] / seqLen)
	assert data.shape[1] == inpSize, "invalid input size"
	return data.reshape(nrow,  seqLen * inpSize)

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

def perfMetric(metric, yActual, yPred, clabels=None):
	"""
	predictive model accuracy metric
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
	"""
	if method == "minmax":
		scaler = preprocessing.MinMaxScaler()
		data = scaler.fit_transform(data)
	elif method == "zscale":
		data = preprocessing.scale(data)	
	else:
		raise ValueError("invalid scaling method")	
	return data

def harmonicNum(n):
	"""
	harmonic number
	"""
	h = 0
	for i in range(1, n+1, 1):
		h += 1.0 / i
	return h
	
def digammaFun(n):
	"""
	figamma function
	"""
	#Euler Mascheroni constant
	ec = 0.577216
	return harmonicNum(n - 1) - ec
			
def getDataPartitions(tdata, types, columns = None):
	"""
	partitions data with the given columns and random split point
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
		
	
class ShiftedDataGenerator:
	"""
	transforms data for distribution shift
	"""
	def __init__(self, types, tdata, addFact, multFact):
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
		

		