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

import os
import sys
from random import randint
import random
import time
import uuid
from datetime import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import logging.handlers
import pickle
from contextlib import contextmanager

tokens = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M",
	"N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]
numTokens = tokens[:10]
alphaTokens = tokens[10:36]
loCaseChars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k","l","m","n","o",
"p","q","r","s","t","u","v","w","x","y","z"]

typeInt = "int"
typeFloat = "float"
typeString = "string"

secInMinute = 60
secInHour = 60 * 60
secInDay = 24 * secInHour
secInWeek = 7 * secInDay
secInYear = 365 * secInDay
secInMonth = secInYear / 12

minInHour = 60
minInDay = 24 * minInHour

ftPerYard = 3
ftPerMile = ftPerYard * 1760


def genID(size):
	"""
	generates ID
	
	Parameters
		size : size of ID
	"""
	id = ""
	for i in range(size):
		id = id + selectRandomFromList(tokens)
	return id

def genIdList(numId, idSize):
	"""
	generate list of IDs
	
	Parameters:
		numId: number of Ids
		idSize: ID size
	"""
	iDs = []
	for i in range(numId):
		iDs.append(genID(idSize))
	return iDs
	
def genNumID(size):
	"""
	generates ID consisting of digits onl
	
	Parameters
		size : size of ID
	"""
	id = ""
	for i in range(size):
		id = id + selectRandomFromList(numTokens)
	return id

def genLowCaseID(size):
	"""
	generates ID consisting of lower case chars
	
	Parameters
		size : size of ID
	"""
	id = ""
	for i in range(size):
		id = id + selectRandomFromList(loCaseChars)
	return id

def genNumIdList(numId, idSize):
	"""
	generate list of numeric IDs
	
	Parameters:
		numId: number of Ids
		idSize: ID size
	"""
	iDs = []
	for i in range(numId):
		iDs.append(genNumID(idSize))
	return iDs

def genNameInitial():
	"""
	generate name initial
	"""
	return selectRandomFromList(alphaTokens) + selectRandomFromList(alphaTokens)

def genPhoneNum(arCode):
	"""
	generates phone number
	
	Parameters
		arCode: area code
	"""
	phNum = genNumID(7)
	return arCode + str(phNum)

def selectRandomFromList(ldata):
	"""
	select an element randomly from a lis
	
	Parameters
		ldata : list data
	"""
	return ldata[randint(0, len(ldata)-1)]

def selectOtherRandomFromList(ldata, cval):
	"""
	select an element randomly from a list excluding the given one
	
	Parameters
		ldata : list data
		cval : value to be excluded
	"""
	nval = selectRandomFromList(ldata)
	while nval == cval:
		nval = selectRandomFromList(ldata)
	return nval
	
def selectRandomSubListFromList(ldata, num):
	"""
	generates random sublist from a list without replacemment
	
	Parameters
		ldata : list data
		num : output list size
	"""
	assertLesser(num, len(ldata), "size of sublist to be sampled greater than or equal to main list")
	i = randint(0, len(ldata)-1)
	sel = ldata[i]
	selSet = {i}
	selList = [sel]
	while (len(selSet) < num):
		i = randint(0, len(ldata)-1)
		if (i not in selSet):
			sel = ldata[i]
			selSet.add(i)
			selList.append(sel)		
	return selList

def selectRandomSubListFromListWithRepl(ldata, num):
	"""
	generates random sublist from a list with replacemment
	
	Parameters
		ldata : list data
		num : output list size

	"""
	return list(map(lambda i : selectRandomFromList(ldata), range(num)))

def selectRandomFromDict(ddata):
	"""
	select an element randomly from a dictionary
	
	Parameters
		ddata : dictionary data
	"""
	dkeys = list(ddata.keys())
	dk = selectRandomFromList(dkeys)
	el = (dk, ddata[dk])
	return el

def setListRandomFromList(ldata, ldataRepl):
	"""
	sets some elents in the first list randomly with elements from the second list
	
	Parameters
		ldata : list data
		ldataRepl : list with replacement data
	"""
	l = len(ldata)
	selSet = set()
	for d in ldataRepl:
		i = randint(0, l-1)
		while i in selSet:
			i = randint(0, l-1)
		ldata[i] = d
		selSet.add(i)
		
def genIpAddress():
	"""
	generates IP address
	"""
	i1 = randint(0,256)
	i2 = randint(0,256)
	i3 = randint(0,256)
	i4 = randint(0,256)
	ip = "%d.%d.%d.%d" %(i1,i2,i3,i4)
	return ip

def curTimeMs():
	"""
	current time in ms
	"""
	return int((datetime.utcnow() - datetime(1970,1,1)).total_seconds() * 1000)

def secDegPolyFit(x1, y1, x2, y2, x3, y3):
	"""
	second deg polynomial 	
	
	Parameters
		x1 : 1st point x
		y1 : 1st point y
		x2 : 2nd point x
		y2 : 2nd point y
		x3 : 3rd point x
		y3 : 3rd point y
	"""
	t = (y1 - y2) / (x1 - x2)
	a = t - (y2 - y3) / (x2 - x3)
	a = a / (x1 - x3)
	b = t - a * (x1 + x2)
	c = y1 - a * x1 * x1 - b * x1
	return (a, b, c)

def range_limit(val, minv, maxv):
	"""
	range limit a value
	
	Parameters
		val : data value
		minv : minimum
		maxv : maximum
	"""
	if (val < minv):
		val = minv
	elif (val > maxv):
		val = maxv
	return val	

def isInRange(val, minv, maxv):
	"""
	checks if within range
	
	Parameters
		val : data value
		minv : minimum
		maxv : maximum
	"""
	return val >= minv and val <= maxv
	
def stripFileLines(filePath, offset):
	"""
	strips number of chars from both ends
	
	Parameters
		filePath : file path
		offset : offset from both ends of  line 
	"""
	fp = open(filePath, "r")
	for line in fp:
		stripped = line[offset:len(line) - 1 - offset]
		print (stripped)
	fp.close()

def genLatLong(lat1, long1, lat2, long2):
	"""
	generate lat log within limits
	
	Parameters
		lat1 : lat of 1st point
		long1 : long of 1st point
		lat2 : lat of 2nd point
		long2 : long of 2nd point
	"""
	lat = lat1 + (lat2 - lat1) * random.random()
	longg = long1 + (long2 - long1) * random.random()
	return (lat, longg)

def geoDistance(lat1, long1, lat2, long2):
	"""
	find geo distance in ft
	
	Parameters
		lat1 : lat of 1st point
		long1 : long of 1st point
		lat2 : lat of 2nd point
		long2 : long of 2nd point
	"""
	latDiff = math.radians(lat1 - lat2)
	longDiff = math.radians(long1 - long2)
	l1 = math.sin(latDiff/2.0)
	l2 = math.sin(longDiff/2.0)
	l3 = math.cos(math.radians(lat1))
	l4 = math.cos(math.radians(lat2))
	a = l1 * l1 + l3 * l4 * l2 * l2
	l5 = math.sqrt(a)
	l6 = math.sqrt(1.0 - a)
	c = 2.0 * math.atan2(l5, l6)
	r = 6371008.8 * 3.280840
	return c * r

def minLimit(val, limit):
	"""
	min limit
	Parameters

	"""
	if (val < limit):
		val = limit
	return val;

def maxLimit(val, limit):
	"""
	max limit
	Parameters

	"""
	if (val > limit):
		val = limit
	return val;

def rangeSample(val, minLim, maxLim):
	"""
	if out side range sample within range
	
	Parameters
		val : value
		minLim : minimum
		maxLim : maximum
	"""
	if val < minLim or val > maxLim:
		val = randint(minLim, maxLim)
	return val

def genRandomIntListWithinRange(size, minLim, maxLim):
	"""
	random unique list of integers within range
	
	Parameters
		size : size of returned list
		minLim : minimum
		maxLim : maximum
	"""
	values = set()
	for i in range(size):
		val = randint(minLim, maxLim)
		while val not in values:
			values.add(val)
	return list(values)

def preturbScalar(value, vrange):
	"""
	preturbs a mutiplicative value within range
	
	Parameters
		value : data value
		vrange : value delta  fraction
	"""
	scale = 1.0 - vrange + 2 * vrange * random.random() 
	return value * scale
	
def preturbScalarAbs(value, vrange):
	"""
	preturbs an absolute value within range
	
	Parameters
		value : data value
		vrange : value delta  absolute

	"""
	delta = - vrange + 2.0 * vrange * random.random() 
	return value + delta

def preturbVector(values, vrange):
	"""
	preturbs a list within range
	
	Parameters
		values : list data
		vrange : value delta  fraction
	"""
	nValues = list(map(lambda va: preturbScalar(va, vrange), values))
	return nValues

def randomShiftVector(values, smin, smax):
	"""
	shifts  a list by a random quanity with a range
	
	Parameters
		values : list data
		smin : samplinf minimum
		smax : sampling maximum
	"""
	shift = np.random.uniform(smin, smax)
	return list(map(lambda va: va + shift, values))

def floatRange(beg, end, incr):
	"""
	generates float range
	
	Parameters
		beg :range begin
		end: range end
		incr : range increment
	"""
	return list(np.arange(beg, end, incr))
	
def shuffle(values, *numShuffles):
	"""
	in place shuffling with swap of pairs
	
	Parameters
		values : list data
		numShuffles : parameters for number of shuffles
	"""
	size = len(values)
	if len(numShuffles) == 0:
		numShuffle = int(size / 2)
	elif len(numShuffles) == 1:
		numShuffle = numShuffles[0]
	else:
		numShuffle = randint(numShuffles[0], numShuffles[1])
	#print("numShuffle {}".format(numShuffle))
	for i in range(numShuffle):
		first = random.randint(0, size - 1)
		second = random.randint(0, size - 1)
		while first == second:
			second = random.randint(0, size - 1)
		tmp = values[first]
		values[first] = values[second]
		values[second] = tmp
		
	
def splitList(itms, numGr):
	"""
	splits a list into sub lists of approximately equal size, with items in sublists randomly chod=sen
	
	Parameters
		itms ; list of values		
		numGr : no of groups
	"""
	tcount = len(itms)
	cItems = list(itms)
	sz = int(len(cItems) / numGr)
	groups = list()
	count = 0
	for i in range(numGr):
		if (i == numGr - 1):
			csz = tcount - count
		else:
			csz = sz + randint(-2, 2)
			count += csz
		gr = list()
		for  j in range(csz):
			it = selectRandomFromList(cItems)
			gr.append(it)	
			cItems.remove(it)	
		groups.append(gr)
	return groups	

def multVector(values, vrange):
	"""
	multiplies a list within value  range
	
	Parameters
		values : list of values
		vrange : fraction of vaue to be used to update
	"""
	scale = 1.0 - vrange + 2 * vrange * random.random()
	nValues = list(map(lambda va: va * scale, values))
	return nValues
	
def weightedAverage(values, weights):
	"""
	calculates weighted average
	
	Parameters
		values : list of values
		weights : list of weights
	"""		
	assert len(values) == len(weights), "values and weights should be same size"
	vw = zip(values, weights)
	wva = list(map(lambda e : e[0] * e[1], vw))
	#wa = sum(x * y for x, y in vw) / sum(weights)
	wav = sum(wva) / sum(weights)
	return wav

def extractFields(line, delim, keepIndices):
	"""
	breaks a line into fields and keeps only specified fileds and returns new line
	
	Parameters
		line ; deli separated string
		delim : delemeter
		keepIndices : list of indexes to fields to be retained
	"""
	items = line.split(delim)
	newLine = []
	for i in keepIndices:
		newLine.append(line[i])
	return delim.join(newLine)

def remFields(line, delim, remIndices):
	"""
	removes fields from delim separated string
	
	Parameters
		line ; delemeter separated string
		delim : delemeter
		remIndices : list of indexes to fields to be removed
	"""
	items = line.split(delim)
	newLine = []
	for i in range(len(items)):
		if not arrayContains(remIndices, i):
			newLine.append(line[i])
	return delim.join(newLine)

def extractList(data, indices):
	"""
	extracts list from another list, given indices
	
	Parameters
		remIndices : list data
		indices : list of indexes to fields to be retained
	"""
	if areAllFieldsIncluded(data, indices):
		exList = data.copy()
		#print("all indices")
	else:
		exList = list()
		le = len(data)
		for i in indices:
			assert i < le , "index {} out of bound {}".format(i, le)
			exList.append(data[i])
	
	return exList
	
def arrayContains(arr, item):
	"""
	checks if array contains an item 
	
	Parameters
		arr : list data
		item : item to search
	"""
	contains = True
	try:
		arr.index(item)
	except ValueError:
		contains = False
	return contains

def strToIntArray(line, delim=","):	
	"""
	int array from delim separated string
	
	Parameters
		line ; delemeter separated string
	"""
	arr = line.split(delim)
	return [int(a) for a in arr]

def strToFloatArray(line, delim=","):	
	"""
	float array from delim separated string
	
	Parameters
		line ; delemeter separated string
	"""
	arr = line.split(delim)
	return [float(a) for a in arr]

def strListOrRangeToIntArray(line):	
	"""
	int array from delim separated string or range
	
	Parameters
		line ; delemeter separated string
	"""
	varr = line.split(",")
	if (len(varr) > 1):
		iarr =  list(map(lambda v: int(v), varr))
	else:
		vrange = line.split(":")
		if (len(vrange) == 2):
			lo = int(vrange[0])
			hi = int(vrange[1])
			iarr = list(range(lo, hi+1))
		else:
			iarr = [int(line)]
	return iarr
				
def toStr(val, precision):
	"""
	converts any type to string	
	
	Parameters
		val : value
		precision ; precision for float value
	"""
	if type(val) == float or type(val) == np.float64 or type(val) == np.float32:
		format = "%" + ".%df" %(precision)
		sVal = format %(val)
	else:
		sVal = str(val)
	return sVal

def toStrFromList(values, precision, delim=","):
	"""
	converts list of any type to delim separated string
	
	Parameters
		values : list data
		precision ; precision for float value
		delim : delemeter
	"""
	sValues = list(map(lambda v: toStr(v, precision), values))
	return delim.join(sValues)

def toIntList(values):
	"""
	convert to int list
	
	Parameters
		values : list data
	"""
	return list(map(lambda va: int(va), values))
		
def toFloatList(values):
	"""
	convert to float list
	
	Parameters
		values : list data

	"""
	return list(map(lambda va: float(va), values))

def toStrList(values, precision=None):
	"""
	convert to string list
	
	Parameters
		values : list data
		precision ; precision for float value
	"""
	return list(map(lambda va: toStr(va, precision), values))
	
def toIntFromBoolean(value):
	"""
	convert to int
	
	Parameters
		value : boolean value
	"""
	ival = 1 if value else 0
	return ival

def typedValue(val, dtype=None):
	"""
	return typed value given string, discovers data type if not specified
	
	Parameters
		val : value
		dtype : data type
	"""
	tVal = None
	
	if dtype is not None:
		if dtype == "num":
			dtype = "int" if dtype.find(".") == -1 else "float"
			
		if dtype == "int":
			tVal = int(val)
		elif dtype == "float":
			tVal = float(val)
		elif dtype == "bool":
			tVal = bool(val)
		else:
			tVal = val
	else:
		if type(val) == str:
			lVal = val.lower()
		
			#int
			done = True
			try:
				tVal = int(val)
			except ValueError:
				done = False
		
			#float
			if not done:	
				done = True
				try:
					tVal = float(val)
				except ValueError:
					done = False
				
			#boolean
			if not done:
				done = True
				if lVal == "true":
					tVal = True
				elif lVal == "false":
					tVal = False
				else:
					done = False
			#None		
			if not done:
				if lVal == "none":
					tVal = None
				else:
					tVal = val
		else:
			tVal = val		
	
	return tVal
	
def getAllFiles(dirPath):
	"""
	get all files recursively
	
	Parameters
		dirPath : directory path
	"""
	filePaths = []
	for (thisDir, subDirs, fileNames) in os.walk(dirPath):
		for fileName in fileNames:
			filePaths.append(os.path.join(thisDir, fileName))
	filePaths.sort()
	return filePaths

def getFileContent(fpath, verbose=False):
	"""
	get file contents in directory
	
	Parameters
		fpath ; directory path
		verbose : verbosity flag
	"""
	# dcument list
	docComplete  = []
	filePaths = getAllFiles(fpath)

	# read files
	for filePath in filePaths:
		if verbose:
			print("next file " + filePath)
		with open(filePath, 'r') as contentFile:
			content = contentFile.read()
			docComplete.append(content)
	return (docComplete, filePaths)

def getOneFileContent(fpath):
	"""
	get one file contents
	
	Parameters
		fpath : file path
	"""
	with open(fpath, 'r') as contentFile:
		docStr = contentFile.read()
	return docStr
	
def getFileLines(dirPath, delim=","):
	"""
	get lines from a file
	
	Parameters
		dirPath : file path
		delim : delemeter
	"""
	lines = list()
	for li in fileRecGen(dirPath, delim):
		lines.append(li)		
	return lines

def getFileSampleLines(dirPath, percen, delim=","):
	"""
	get sampled lines from a file
	
	Parameters
		dirPath : file path
		percen : sampling percentage
		delim : delemeter
	"""
	lines = list()
	for li in fileRecGen(dirPath, delim):
		if randint(0, 100) < percen:
			lines.append(li)		
	return lines

def getFileColumnAsString(dirPath, index, delim=","):
	"""
	get string column from a file
	
	Parameters
		dirPath : file path
		index : index
		delim : delemeter
	"""
	fields = list()
	for rec in fileRecGen(dirPath, delim):
		fields.append(rec[index])	
	#print(fields)	
	return fields

def getFileColumnsAsString(dirPath, indexes, delim=","):
	"""
	get multiple string columns from a file
	
	Parameters
		dirPath : file path
		indexes : indexes of columns
		delim : delemeter

	"""
	nindex = len(indexes)
	columns = list(map(lambda i : list(), range(nindex)))
	for rec in fileRecGen(dirPath, delim):
		for i in range(nindex):
			columns[i].append(rec[indexes[i]])	
	return columns

def getFileColumnAsFloat(dirPath, index, delim=","):
	"""
	get float fileds from a file
	
	Parameters
		dirPath : file path
		index : index
		delim : delemeter

	"""
	#print("{}  {}".format(dirPath, index))
	fields = getFileColumnAsString(dirPath, index, delim)
	return list(map(lambda v:float(v), fields))
	
def getFileColumnAsInt(dirPath, index, delim=","):
	"""
	get float fileds from a file
	
	Parameters
		dirPath : file path
		index : index
		delim : delemeter
	"""
	fields = getFileColumnAsString(dirPath, index, delim)
	return list(map(lambda v:int(v), fields))

def getFileAsIntMatrix(dirPath, columns, delim=","):
	"""
	extracts int matrix from csv file given column indices with each row being  concatenation of 
	extracted column values row size = num of columns
	
	Parameters
		dirPath : file path
		columns : indexes of columns
		delim : delemeter
	"""
	mat = list()
	for rec in  fileSelFieldsRecGen(dirPath, columns, delim):
		mat.append(asIntList(rec))
	return mat

def getFileAsFloatMatrix(dirPath, columns, delim=","):
	"""
	extracts float matrix from csv file given column indices with each row being concatenation of  
	extracted column values row size = num of columns

	Parameters
		dirPath : file path
		columns : indexes of columns
		delim : delemeter
	"""
	mat = list()
	for rec in  fileSelFieldsRecGen(dirPath, columns, delim):
		mat.append(asFloatList(rec))
	return mat
	
def getFileAsFloatColumn(dirPath):
	"""
	grt float list from a file with one float per row

	Parameters
		dirPath : file path
	"""
	flist = list()
	for rec in fileRecGen(dirPath, None):
		flist.append(float(rec))
	return flist

def getFileAsFiltFloatMatrix(dirPath, filt, columns, delim=","):
	"""
	extracts float matrix from csv file given row filter and column indices with each row being 
	concatenation of  extracted column values row size = num of columns

	Parameters
		dirPath : file path
		columns : indexes of columns
		filt : row filter lambda
		delim : delemeter

	"""
	mat = list()
	for rec in  fileFiltSelFieldsRecGen(dirPath, filt, columns, delim):
		mat.append(asFloatList(rec))
	return mat

def getFileAsTypedRecords(dirPath, types, delim=","):
	"""
	extracts typed records from csv file with each row being concatenation of  
	extracted column values 

	Parameters
		dirPath : file path
		types : data types
		delim : delemeter
	"""
	(dtypes, cvalues) = extractTypesFromString(types)	
	tdata = list()
	for rec in  fileRecGen(dirPath, delim):
		trec = list()
		for index, value in enumerate(rec):
			value = __convToTyped(index, value, dtypes)
			trec.append(value)
		tdata.append(trec)
	return tdata

	
def getFileColsAsTypedRecords(dirPath, columns, types, delim=","):
	"""
	extracts typed records from csv file given column indices with each row being concatenation of  
	extracted column values 

	Parameters
	Parameters
		dirPath : file path
		columns : column indexes
		types : data types
		delim : delemeter
	"""
	(dtypes, cvalues) = extractTypesFromString(types)	
	tdata = list()
	for rec in  fileSelFieldsRecGen(dirPath, columns, delim):
		trec = list()
		for indx, value in enumerate(rec):
			tindx = columns[indx]
			value = __convToTyped(tindx, value, dtypes)
			trec.append(value)
		tdata.append(trec)
	return tdata

def getFileColumnsMinMax(dirPath, columns, dtype, delim=","):
	"""
	extracts numeric matrix from csv file given column indices. For each column return min and max

	Parameters
		dirPath : file path
		columns : column indexes
		dtype : data type
		delim : delemeter
	"""
	dtypes = list(map(lambda c : str(c) + ":" + dtype, columns))
	dtypes = ",".join(dtypes)
	#print(dtypes)
	
	tdata = getFileColsAsTypedRecords(dirPath, columns, dtypes, delim)
	minMax = list()
	ncola = len(tdata[0])
	ncole = len(columns)
	assertEqual(ncola, ncole, "actual no of columns different from expected")
	
	for ci in range(ncole):	
		vmin = sys.float_info.max
		vmax = sys.float_info.min
		for r in tdata:
			cv = r[ci]
			vmin = cv if cv < vmin else vmin
			vmax = cv if cv > vmax else vmax
		mm = (vmin, vmax, vmax - vmin)
		minMax.append(mm)

	return minMax


def getRecAsTypedRecord(rec, types, delim=None):
	"""
	converts record to  typed records 

	Parameters
		rec : delemeter separate string or list of string
		types : field  data types
		delim : delemeter
	"""	
	if delim is not None:
		rec = rec.split(delim)
	(dtypes, cvalues) = extractTypesFromString(types)	
	#print(types)
	#print(dtypes)
	trec = list()
	for ind, value in enumerate(rec):
		tvalue = __convToTyped(ind, value, dtypes)
		trec.append(tvalue)
	return trec
		
def __convToTyped(index, value, dtypes):
	"""
	convert to typed value 

	Parameters
		index : index in type list
		value : data value
		dtypes : data type list
	"""
	#print(index, value)
	dtype = dtypes[index]
	tvalue = value
	if dtype == "int":
		tvalue = int(value)
	elif dtype == "float":
		tvalue = float(value)
	return tvalue
	
	

def extractTypesFromString(types):
	"""
	extracts column data types and set values for categorical variables 

	Parameters
		types : encoded type information
	"""
	ftypes = types.split(",")
	dtypes = dict()
	cvalues = dict()
	for ftype in ftypes:
		items = ftype.split(":") 
		cindex = int(items[0])
		dtype = items[1]
		dtypes[cindex] = dtype
		if len(items) == 3:
			sitems = items[2].split()
			cvalues[cindex] = sitems
	return (dtypes, cvalues)
	
def getMultipleFileAsInttMatrix(dirPathWithCol,  delim=","):
	"""
	extracts int matrix from from csv files given column index for each file. 
	num of columns  = number of rows in each file and num of rows = number of files

	Parameters
		dirPathWithCol: list of file path and collumn index pair
		delim : delemeter
	"""
	mat = list()
	minLen = -1
	for path, col in dirPathWithCol:
		colVals = getFileColumnAsInt(path, col, delim)
		if minLen < 0 or len(colVals) < minLen:
			minLen = len(colVals)
		mat.append(colVals)

	#make all same length
	mat = list(map(lambda li:li[:minLen], mat))	
	return mat

def getMultipleFileAsFloatMatrix(dirPathWithCol,  delim=","):
	"""
	extracts float matrix from from csv files given column index for each file. 
	num of columns  = number of rows in each file and num of rows = number of files

	Parameters
		dirPathWithCol: list of file path and collumn index pair
		delim : delemeter
	"""
	mat = list()
	minLen = -1
	for path, col in dirPathWithCol:
		colVals = getFileColumnAsFloat(path, col, delim)
		if minLen < 0 or len(colVals) < minLen:
			minLen = len(colVals)
		mat.append(colVals)
	
	#make all same length
	mat = list(map(lambda li:li[:minLen], mat))	
	return mat

def writeStrListToFile(ldata, filePath, delem=","):
	"""
	writes list of dlem separated string or list of list of string to afile
	
	Parameters
		ldata : list data
		filePath : file path
		delim : delemeter
	"""
	with open(filePath, "w") as fh:
		for r in ldata:
			if type(r) == list:
				r = delem.join(r)
			fh.write(r + "\n")

def writeFloatListToFile(ldata, prec, filePath):
	"""
	writes float list to file, one value per line
	
	Parameters
		ldata : list data
		prec : precision
		filePath : file path
	"""
	with open(filePath, "w") as fh:
		for d in ldata:
			fh.write(formatFloat(prec, d) + "\n")

	
def takeFirst(elems):
	"""
	return fisrt item

	Parameters
		elems : list of data 
	"""
	return elems[0]

def takeSecond(elems):
	"""
	return 2nd element

	Parameters
		elems : list of data 
	"""
	return elems[1]

def takeThird(elems):
	"""
	returns 3rd element

	Parameters
		elems : list of data 
	"""
	return elems[2]

def addToKeyedCounter(dCounter, key, count=1):
	"""
	add to to keyed counter

	Parameters
		dCounter : dictionary of counters
		key : dictionary key
		count : count to add
	"""
	curCount = dCounter.get(key, 0)
	dCounter[key] = curCount + count

def incrKeyedCounter(dCounter, key):
	"""
	increment keyed counter

	Parameters
		dCounter : dictionary of counters
		key : dictionary key
	"""
	addToKeyedCounter(dCounter, key, 1)

def appendKeyedList(dList, key, elem):
	"""
	keyed list

	Parameters
		dList : dictionary of lists
		key : dictionary key
		elem : value to append
	"""
	curList = dList.get(key, [])
	curList.append(elem)
	dList[key] = curList

def isNumber(st):
	"""
	Returns True is string is a number

	Parameters
		st : string value
	"""
	return st.replace('.','',1).isdigit()

def removeNan(values):
	"""
	removes nan from list

	Parameters
		values : list data
	"""
	return list(filter(lambda v: not math.isnan(v), values))
	
def fileRecGen(filePath, delim = ","):
	"""
	file record generator

	Parameters
		filePath ; file path
		delim : delemeter
	"""
	with open(filePath, "r") as fp:
		for line in fp:	
			line = line[:-1]
			if delim is not None:
				line = line.split(delim)
			yield line

def fileSelFieldsRecGen(dirPath, columns, delim=","):
	"""
	file record generator given column indices 

	Parameters
		filePath ; file path
		columns : column indexes as int array or coma separated string
		delim : delemeter
	"""
	if type(columns) == str:
		columns = strToIntArray(columns, delim)
	for rec in fileRecGen(dirPath, delim):
		extracted = extractList(rec, columns)
		yield extracted

def fileFiltRecGen(filePath, filt, delim = ","):
	"""
	file record generator with  row filter applied

	Parameters
		filePath ; file path
		filt : row filter
		delim : delemeter
	"""
	with open(filePath, "r") as fp:
		for line in fp:	
			line = line[:-1]
			if delim is not None:
				line = line.split(delim)
			if filt(line):
				yield line

def fileFiltSelFieldsRecGen(filePath, filt, columns, delim = ","):
	"""
	file record generator with  row and column filter applied

	Parameters
		filePath ; file path
		filt : row filter
		columns : column indexes as int array or coma separated string
		delim : delemeter
	"""
	columns = strToIntArray(columns, delim)
	with open(filePath, "r") as fp:
		for line in fp:	
			line = line[:-1]
			if delim is not None:
				line = line.split(delim)
			if filt(line):
				selected = extractList(line, columns)
				yield selected

def fileTypedRecGen(filePath, ftypes, delim = ","):
	"""
	file typed record generator

	Parameters
		filePath ; file path
		ftypes : list of field types
		delim : delemeter
	"""
	with open(filePath, "r") as fp:
		for line in fp:	
			line = line[:-1]
			line = line.split(delim)
			for i in range(0, len(ftypes), 2):
				ci = ftypes[i]
				dtype = ftypes[i+1]
				assertLesser(ci, len(line), "index out of bound")
				if dtype == "int":
					line[ci] = int(line[ci])
				elif dtype == "float":
					line[ci] = float(line[ci])
				else:
					exitWithMsg("invalid data type")
			yield line

def fileMutatedFieldsRecGen(dirPath, mutator, delim=","):
	"""
	file record generator with some columns mutated 

	Parameters
		dirPath ; file path
		mutator : row field mutator
		delim : delemeter
	"""
	for rec in fileRecGen(dirPath, delim):
		mutated = mutator(rec)
		yield mutated

def tableSelFieldsFilter(tdata, columns):
	"""
	gets tabular data for selected columns 

	Parameters
		tdata : tabular data
		columns : column indexes
	"""
	if areAllFieldsIncluded(tdata[0], columns):
		ntdata = tdata
	else:
		ntdata = list()
		for rec in tdata:
			#print(rec)
			#print(columns)
			nrec = extractList(rec, columns)
			ntdata.append(nrec)
	return ntdata	
	

def areAllFieldsIncluded(ldata, columns):
	"""
	return True id all indexes are in the columns

	Parameters
		ldata : list data
		columns : column indexes
	"""
	return list(range(len(ldata))) == columns
	
def asIntList(items):
	"""
	returns int list

	Parameters
		items : list data
	"""
	return [int(i) for i in items]
			
def asFloatList(items):
	"""
	returns float list

	Parameters
		items : list data
	"""
	return [float(i) for i in items]

def pastTime(interval, unit):
	"""
	current and past time

	Parameters
		interval : time interval
		unit: time unit
	"""
	curTime = int(time.time())
	if unit == "d":
		pastTime = curTime - interval * secInDay
	elif unit == "h":
		pastTime = curTime - interval * secInHour
	elif unit == "m":
		pastTime = curTime - interval * secInMinute
	else:
		raise ValueError("invalid time unit " + unit)
	return (curTime, pastTime)

def minuteAlign(ts):
	"""
	minute aligned time	

	Parameters
		ts : time stamp in sec
	"""
	return int((ts / secInMinute)) * secInMinute

def multMinuteAlign(ts, min):
	"""
	multi minute aligned time	

	Parameters
		ts : time stamp in sec
		min : minute value
	"""
	intv = secInMinute * min
	return int((ts / intv)) * intv

def hourAlign(ts):
	"""
	hour aligned time	

	Parameters
		ts : time stamp in sec
	"""
	return int((ts / secInHour)) * secInHour
	
def hourOfDayAlign(ts, hour):
	"""
	hour of day aligned time	

	Parameters
		ts : time stamp in sec
		hour : hour of day
	"""
	day = int(ts / secInDay)
	return (24 * day + hour) * secInHour

def dayAlign(ts):
	"""
	day aligned time	

	Parameters
		ts : time stamp in sec
	"""
	return int(ts / secInDay) * secInDay

def timeAlign(ts, unit):
	"""
	boundary alignment of time

	Parameters
		ts : time stamp in sec
		unit : unit of time
	"""
	alignedTs = 0
	if unit == "s":
		alignedTs = ts
	elif unit == "m":
		alignedTs = minuteAlign(ts)
	elif unit == "h":
		alignedTs = hourAlign(ts)
	elif unit == "d":
		alignedTs = dayAlign(ts)
	else:
		raise ValueError("invalid time unit")
	return 	alignedTs

def monthOfYear(ts):
	"""
	month of year

	Parameters
		ts : time stamp in sec
	"""
	rem = ts % secInYear
	dow = int(rem / secInMonth)
	return dow
		
def dayOfWeek(ts):
	"""
	day of week

	Parameters
		ts : time stamp in sec
	"""
	rem = ts % secInWeek
	dow = int(rem / secInDay)
	return dow

def hourOfDay(ts):
	"""
	hour of day

	Parameters
		ts : time stamp in sec
	"""
	rem = ts % secInDay
	hod = int(rem / secInHour)
	return hod
	
def processCmdLineArgs(expectedTypes, usage):
	"""
	process command line args and returns args as typed values

	Parameters
		expectedTypes : expected data types of arguments
		usage : usage message string
	"""
	args = []
	numComLineArgs = len(sys.argv)
	numExpected = len(expectedTypes)
	if (numComLineArgs - 1 == len(expectedTypes)):
		try:
			for i in range(0, numExpected):
				if (expectedTypes[i] == typeInt):
					args.append(int(sys.argv[i+1]))
				elif (expectedTypes[i] == typeFloat):
					args.append(float(sys.argv[i+1]))
				elif (expectedTypes[i] == typeString):
					args.append(sys.argv[i+1])
		except ValueError:
			print ("expected number of command line arguments found but there is type mis match")
			sys.exit(1)
	else:
		print ("expected number of command line arguments not found")
		print (usage)
		sys.exit(1)
	return args
	
def mutateString(val, numMutate, ctype):
	"""
	mutate string multiple times

	Parameters
		val : string value
		numMutate : num of mutations
		ctype : type of character to mutate with
	"""
	mutations = set()
	count = 0
	while count < numMutate:
		j = randint(0, len(val)-1)
		if j not in mutations:
			if ctype == "alpha":
				ch = selectRandomFromList(alphaTokens)
			elif ctype == "num":
				ch = selectRandomFromList(numTokens)
			elif ctype == "any":
				ch = selectRandomFromList(tokens)
			val = val[:j] + ch + val[j+1:]
			mutations.add(j)
			count += 1
	return val

def mutateList(values, numMutate, vmin, vmax):
	"""
	mutate list multiple times

	Parameters
		values : list value
		numMutate : num of mutations
		vmin : minimum of value range
		vmax : maximum of value range
	"""
	mutations = set()
	count = 0
	while count < numMutate:
		j = randint(0, len(values)-1)
		if j not in mutations:
			values[j] = np.random.uniform(vmin, vmax)
			count += 1
	return values		
	

def swap(values, first, second):
	"""
	swap two elements

	Parameters
		values : list value
		first : first swap position
		second : second swap position
	"""
	t = values[first]
	values[first] = values[second]	
	values[second] = t

def swapBetweenLists(values1, values2):
	"""
	swap two elements between 2 lists

	Parameters
		values1 : first list of values
		values2 : second list of values
	"""
	p1 = randint(0, len(values1)-1)
	p2 = randint(0, len(values2)-1)
	tmp = values1[p1]	
	values1[p1] = values2[p2]
	values2[p2] = tmp

def safeAppend(values, value):
	"""
	append only if not None

	Parameters
		values : list value
		value : value to append
	"""
	if value is not None:
		values.append(value)

def getAllIndex(ldata, fldata):
	"""
	get ALL indexes of list elements

	Parameters
		ldata : list data to find index in
		fldata : list data for values for index look up
	"""
	return list(map(lambda e : fldata.index(e), ldata))

def findIntersection(lOne, lTwo):
	"""
	find intersection elements between 2 lists

	Parameters
		lOne : first list of data
		lTwo : second list of data
	"""
	sOne = set(lOne)
	sTwo = set(lTwo)
	sInt = sOne.intersection(sTwo)
	return list(sInt)

def isIntvOverlapped(rOne, rTwo):
	"""
	checks overlap between 2 intervals

	Parameters
		rOne : first interval boundaries
		rTwo : second interval boundaries
	"""
	clear = rOne[1] <=  rTwo[0] or rOne[0] >=  rTwo[1] 
	return not clear

def isIntvLess(rOne, rTwo):
	"""
	checks if first iterval is less than second

	Parameters
		rOne : first interval boundaries
		rTwo : second interval boundaries
	"""
	less = rOne[1] <=  rTwo[0] 
	return less

def findRank(e, values):
	"""
	find rank of value in a list

	Parameters
		e : value to compare with
		values : list data
	"""
	count =  1
	for ve in values:
		if ve < e:
			count += 1
	return count

def findRanks(toBeRanked, values):
	"""
	find ranks of values in one list in another list

	Parameters
		toBeRanked : list of values for which ranks are found
		values : list in which rank is found : 
	"""
	return list(map(lambda e: findRank(e, values), toBeRanked))
	
def formatFloat(prec, value, label = None):
	"""
	formats a float with optional label

	Parameters
		prec : precision
		value : data value
		label : label for data
	"""
	st = (label + " ") if label else ""
	formatter = "{:." + str(prec) + "f}" 
	return st + formatter.format(value)
	
def formatAny(value, label = None):
	"""
	formats any obkect with optional label

	Parameters
		value : data value
		label : label for data
	"""
	st = (label + " ") if label else ""
	return st + str(value)

def printList(values):
	"""
	pretty print list

	Parameters
		values : list of values
	"""
	for v in values:
		print(v)

def printMap(values, klab, vlab, precision, offset=16):
	"""
	pretty print hash map

	Parameters
		values : dictionary of values
		klab : label for key
		vlab : label for value
		precision : precision
		offset : left justify offset
	"""
	print(klab.ljust(offset, " ") + vlab)
	for k in values.keys():
		v = values[k]
		ks = toStr(k, precision).ljust(offset, " ")
		vs = toStr(v, precision)
		print(ks +  vs)
		
def printPairList(values, lab1, lab2, precision, offset=16):
	"""
	pretty print list of pairs

	Parameters
		values : dictionary of values
		lab1 : first label
		lab2 : second label
		precision : precision
		offset : left justify offset
	"""
	print(lab1.ljust(offset, " ") + lab2)
	for (v1, v2) in values:
		sv1 = toStr(v1, precision).ljust(offset, " ")
		sv2 = toStr(v2, precision)
		print(sv1 + sv2)

def createMap(*values):
	"""
	create disctionary with results

	Parameters
		values : sequence of key value pairs
	"""
	result = dict()
	for i in range(0, len(values), 2):
		result[values[i]] = values[i+1]
	return result

def getColMinMax(table, col):
	"""
	return min, max values of a column

	Parameters
		table : tabular data
		col : column index
	"""
	vmin = None
	vmax = None
	for rec in table:
		value = rec[col]
		if vmin is None:
			vmin = value
			vmax = value
		else:
			if value < vmin:
				vmin = value
			elif value > vmax:
				vmax = value
	return (vmin, vmax, vmax - vmin)
			
def createLogger(name, logFilePath, logLevName):
	"""
	creates logger

	Parameters
		name : logger name
		logFilePath : log file path
		logLevName : log level
	"""
	logger = logging.getLogger(name)
	fHandler = logging.handlers.RotatingFileHandler(logFilePath, maxBytes=1048576, backupCount=4)
	logLev = logLevName.lower()
	if logLev == "debug":
		logLevel = logging.DEBUG
	elif logLev == "info":
		logLevel = logging.INFO
	elif logLev == "warning":
		logLevel = logging.WARNING
	elif logLev == "error":
		logLevel = logging.ERROR
	elif logLev == "critical":
		logLevel = logging.CRITICAL
	else:
		raise ValueError("invalid log level name " + logLevelName)
	fHandler.setLevel(logLevel)
	fFormat = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
	fHandler.setFormatter(fFormat)
	logger.addHandler(fHandler)
	logger.setLevel(logLevel)
	return logger

@contextmanager
def suppressStdout():
	"""
	suppress stdout

	Parameters

	"""
	with open(os.devnull, "w") as devnull:
		oldStdout = sys.stdout
		sys.stdout = devnull
		try:  
			yield
		finally:
			sys.stdout = oldStdout
			
def exitWithMsg(msg):
	"""
	print message and exit

	Parameters
		msg : message
	"""
	print(msg + " -- quitting")
	sys.exit(0)

def drawLine(data, yscale=None):
	"""
	line plot

	Parameters
		data : list data
		yscale : y axis scale
	"""
	plt.plot(data)
	if yscale:
		step = int(yscale / 10)
		step = int(step / 10) * 10
		plt.yticks(range(0, yscale, step))
	plt.show()

def drawPlot(x, y, xlabel, ylabel):
	"""
	line plot

	Parameters
		x : x values
		y : y values
		xlabel : x axis label
		ylabel : y axis label
	"""
	plt.plot(x,y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

def drawPairPlot(x, y1, y2, xlabel,ylabel, y1label, y2label):
	"""
	line plot of 2 lines

	Parameters
		x : x values
		y1 : first y values
		y2 : second y values
		xlabel : x labbel
		ylabel : y label
		y1label : first plot label
		y2label : second plot label
	"""
	plt.plot(x, y1, label = y1label)
	plt.plot(x, y2, label = y2label)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.show()

def drawHist(ldata, myTitle, myXlabel, myYlabel, nbins=10):
	"""
	draw histogram

	Parameters
		ldata : list data
		myTitle : title
		myXlabel : x label
		myYlabel : y label 
		nbins : num of bins
	"""
	plt.hist(ldata, bins=nbins, density=True)
	plt.title(myTitle)
	plt.xlabel(myXlabel)
	plt.ylabel(myYlabel)
	plt.show()	
	
def saveObject(obj, filePath):
	"""
	saves an object

	Parameters
		obj : object
		filePath : file path for saved object
	"""
	with open(filePath, "wb") as outfile:
		pickle.dump(obj,outfile)
	
def restoreObject(filePath):
	"""
	restores an object

	Parameters
		filePath : file path to restore object from
	"""
	with open(filePath, "rb") as infile:
		obj = pickle.load(infile)
	return obj

def isNumeric(data):
	"""
	true if all elements int or float

	Parameters
		data : numeric data list
	"""
	if type(data) == list or type(data) == np.ndarray:
		col = pd.Series(data)
	else:
		col = data
	return col.dtype == np.int32 or col.dtype == np.int64 or col.dtype == np.float32 or col.dtype == np.float64

def isInteger(data):
	"""
	true if all elements int 

	Parameters
		data : numeric data list
	"""
	if type(data) == list or type(data) == np.ndarray:
		col = pd.Series(data)
	else:
		col = data
	return col.dtype == np.int32 or col.dtype == np.int64

def isFloat(data):
	"""
	true if all elements  float

	Parameters
		data : numeric data list
	"""
	if type(data) == list or type(data) == np.ndarray:
		col = pd.Series(data)
	else:
		col = data
	return col.dtype == np.float32 or col.dtype == np.float64

def isBinary(data):
	"""
	true if all elements either 0 or 1

	Parameters
		data : binary data
	"""
	re = next((d for d in data if not (type(d) == int and (d == 0 or d == 1))), None)
	return (re is None)
	
def isCategorical(data):
	"""
	true if all elements int or string

	Parameters
		data : data value
	"""
	re = next((d for d in data if not (type(d) == int or type(d) == str)), None)
	return (re is None)

def assertEqual(value, veq, msg):
	"""
	assert equal to

	Parameters
		value : value
		veq : value to be equated with
		msg : error msg
	"""
	assert value == veq , msg

def assertGreater(value, vmin, msg):
	"""
	assert greater than 

	Parameters
		value : value
		vmin : minimum value
		msg : error msg
	"""
	assert value > vmin , msg

def assertGreaterEqual(value, vmin, msg):
	"""
	assert greater than 

	Parameters
		value : value
		vmin : minimum value
		msg : error msg
	"""
	assert value >= vmin , msg

def assertLesser(value, vmax, msg):
	"""
	assert less than

	Parameters
		value : value
		vmax : maximum value
		msg : error msg
	"""
	assert value < vmax , msg

def assertLesserEqual(value, vmax, msg):
	"""
	assert less than

	Parameters
		value : value
		vmax : maximum value
		msg : error msg
	"""
	assert value <= vmax , msg

def assertWithinRange(value, vmin, vmax, msg):
	"""
	assert within range

	Parameters
		value : value
		vmin : minimum value
		vmax : maximum value
		msg : error msg
	"""
	assert value >= vmin and value <= vmax, msg
		
def assertInList(value, values, msg):
	"""
	assert contains in a list

	Parameters
		value ; balue to check for inclusion
		values : list data
		msg : error msg
	"""
	assert value in values, msg

def maxListDist(l1, l2):
	"""
	maximum list element difference between 2 lists

	Parameters
		l1 : first list data
		l2 : second list data
	"""
	dist = max(list(map(lambda v : abs(v[0] - v[1]), zip(l1, l2))))	
	return dist

def fileLineCount(fPath):
	""" 
	number of lines ina file 

	Parameters
		fPath : file path
	"""
	with open(fPath) as f:
		for i, li in enumerate(f):
			pass
	return (i + 1)

def getAlphaNumCharCount(sdata):
	""" 
	number of alphabetic and numeric charcters in a string 

	Parameters
		sdata : string data
	"""
	acount = 0
	ncount = 0
	ocount = 0
	assertEqual(type(sdata), str, "input must be string")
	for c in sdata:
		if c.isnumeric():
			ncount += 1
		elif c.isalpha():
			acount += 1
		else:
			ocount += 1
	r = (acount, ncount, ocount)
	return r	
			
class StepFunction:
	"""
	step function

	Parameters

	"""
	def __init__(self,  *values):
		"""
		initilizer
		
		Parameters
			values : list of tuples, wich each tuple containing 2 x values and corresponding y value
		"""
		self.points = values
	
	def find(self, x):
		"""
		finds step function value
		
		Parameters
			x : x value
		"""
		found = False
		y = 0
		for p in self.points:
			if (x >= p[0] and x < p[1]):
				y = p[2]
				found = True
				break
		
		if not found:
			l = len(self.points)
			if (x < self.points[0][0]):
				y = self.points[0][2]
			elif (x > self.points[l-1][1]):
				y = self.points[l-1][2]
		return y
		
	 
class DummyVarGenerator:
	"""
	dummy variable generator for categorical variable
	"""
	def __init__(self,  rowSize, catValues, trueVal, falseVal, delim=None):
		"""
		initilizer
		
		Parameters
			rowSize : row size
			catValues : dictionary with field index as key and list of categorical values as value
			trueVal : true value, typically "1"
			falseval : false value , typically "0"
			delim : field delemeter
		"""
		self.rowSize = rowSize
		self.catValues = catValues
		numCatVar = len(catValues)
		colCount = 0
		for v in self.catValues.values():
			colCount += len(v)
		self.newRowSize = rowSize - numCatVar + colCount
		#print ("new row size {}".format(self.newRowSize))
		self.trueVal = trueVal
		self.falseVal = falseVal
		self.delim = delim
	
	def processRow(self, row):	
		"""
		encodes categorical variables, returning as delemeter separate dstring or list
		
		Parameters
			row : row either delemeter separated string or list
		"""
		if self.delim is not None:
			rowArr = row.split(self.delim)
			msg = "row does not have expected number of columns found " + str(len(rowArr)) + " expected " + str(self.rowSize)
			assert len(rowArr) == self.rowSize, msg
		else:
			rowArr = row
			
		newRowArr = []
		for i in range(len(rowArr)):
			curVal = rowArr[i]
			if (i in self.catValues):
				values = self.catValues[i]
				for val in values:
					if val == curVal:
						newVal = self.trueVal
					else:
						newVal = self.falseVal
					newRowArr.append(newVal)
			else:
				newRowArr.append(curVal)
		assert len(newRowArr) == self.newRowSize, "invalid new row size " + str(len(newRowArr)) + " expected " + str(self.newRowSize)
		encRow = self.delim.join(newRowArr) if self.delim is not None else newRowArr
		return encRow
		
		
