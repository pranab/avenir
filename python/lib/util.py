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

import os
import sys
from random import randint
import random
import time
import uuid
from datetime import datetime
import math
import logging
from contextlib import contextmanager

tokens = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M",
	"N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]
numTokens = tokens[:10]
alphaTokens = tokens[10:36]

typeInt = "int"
typeFloat = "float"
typeString = "string"

secInMinute = 60
secInHour = 60 * 60
secInDay = 24 * secInHour
secInWeek = 7 * secInDay
secInYear = 365 * secInDay
secInMonth = secInYear / 12

def genID(len):
	"""
	generates ID
	"""
	id = ""
	for i in range(len):
		id = id + selectRandomFromList(tokens)
	return id

def genIdList(numId, idSize):
	"""
	generate list of IDs
	"""
	iDs = []
	for i in range(numId):
		iDs.append(genID(idSize))
	return iDs
	
def genNumID(len):
	"""
	generates ID consisting of digits onl
	"""
	id = ""
	for i in range(len):
		id = id + selectRandomFromList(numTokens)
	return id

def genNumIdList(numId, idSize):
	"""
	generate list of numeric IDs
	"""
	iDs = []
	for i in range(numId):
		iDs.append(genNumID(idSize))
	return iDs

def genPhoneNum(arCode):
	"""
	generates phone number
	"""
	phNum = genNumID(7)
	return arCode + str(phNum)

def selectRandomFromList(list):
	"""
	select an element randomly from a lis
	"""
	return list[randint(0, len(list)-1)]

def selectRandomSubListFromList(list, num):
	"""
	generates random sublist from a list
	"""
	sel = selectRandomFromList(list)
	selSet = {sel}
	selList = [sel]
	while (len(selSet) < num):
		sel = selectRandomFromList(list)
		if (sel not in selSet):
			selSet.add(sel)
			selList.append(sel)		
	return selList

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
	"""
	t = (y1 - y2) / (x1 - x2)
	a = t - (y2 - y3) / (x2 - x3)
	a = a / (x1 - x3)
	b = t - a * (x1 + x2)
	c = y1 - a * x1 * x1 - b * x1
	return (a, b, c)

def range_limit(val, min, max):
	"""
	range limit
	"""
	if (val < min):
		val = min
	elif (val > max):
		val = max
	return val	

def stripFileLines(filePath, offset):
	"""
	strips number of chars from both ends
	"""
	fp = open(filePath, "r")
	for line in fp:
		stripped = line[offset:len(line) - 1 - offset]
		print (stripped)
	fp.close()

def genLatLong(lat1, long1, lat2, long2):
	"""
	generate lat log within limits
	"""
	lat = lat1 + (lat2 - lat1) * random.random()
	longg = long1 + (long2 - long1) * random.random()
	return (lat, longg)

def geoDistance(lat1, long1, lat2, long2):
	"""
	find geo distance
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
	"""
	if (val < limit):
		val = limit
	return val;

def maxLimit(val, limit):
	"""
	max limit
	"""
	if (val > limit):
		val = limit
	return val;

def rangeSample(val, minLim, maxLim):
	"""
	if out side range sample within range
	"""
	if val < minLim or val > maxLim:
		val = randint(minLim, maxLim)
	return val

def genRandomIntListWithinRange(size, minLim, maxLim):
	"""
	random unique list of integers within range
	"""
	values = set()
	for i in range(size):
		val = randint(minLim, maxLim)
		while val not in values:
			values.add(val)
	return list(values)

def preturbScalar(value, range):
	"""
	preturbs a mutiplicative value within range
	"""
	scale = 1.0 - range + 2 * range * random.random() 
	return value * scale
	
def preturbScalarAbs(value, range):
	"""
	preturbs an absolute value within range
	"""
	delta = - range + 2.0 * range * random.random() 
	return value + delta

def preturbVector(values, range):
	"""
	preturbs a list within range
	"""
	nValues = list(map(lambda va: preturbScalar(va, range), values))
	return nValues

def splitList(itms, numGr):
	"""
	splits a list into sub lists
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

def multVector(values, range):
	"""
	multiplies a list within range
	"""
	scale = 1.0 - range + 2 * range * random.random()
	nValues = list(map(lambda va: va * scale, values))
	return nValues
			
def extractFields(line, delim, keepIndices):
	"""
	breaks a line into fields and keeps only specified fileds and returns new line
	"""
	items = line.split(delim)
	newLine = []
	for i in keepIndices:
		newLine.append(line[i])
	return delim.join(newLine)

def remFields(line, delim, remIndices):
	"""
	removes fields from delim separated string
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
	"""
	exList = list()
	for i in indices:
		exList.append(data[i])
	return exList
	
def arrayContains(arr, item):
	"""
	checks if array contains an item 
	"""
	contains = True
	try:
		arr.index(item)
	except ValueError:
		contains = False
	return contains

def strToIntArray(line, delim):	
	"""
	int array from delim separated string
	"""
	arr = line.split(delim)
	return [int(a) for a in arr]

def strToFloatArray(line, delim):	
	"""
	float array from delim separated string
	"""
	arr = line.split(delim)
	return [float(a) for a in arr]

def strListOrRangeToIntArray(line):	
	"""
	int array from delim separated string or range
	"""
	varr = line.split(",")
	if (len(varr) > 1):
		iarr =  list(map(lambda v: int(v), varr))
	else:
		vrange = line.split(":")
		if (len(vrange) == 2):
			lo = int(varr[0])
			hi = int(varr[1])
			iarr = list(range(lo, hi+1))
		else:
			raise ValueError("failed to generate list")
	return iarr
				
def toStr(val, precision):
	"""
	converts any type to string	
	"""
	if type(val) is float:
		format = "%" + ".%df" %(precision)
		sVal = format %(val)
	else:
		sVal = str(val)
	return sVal

def toStrFromList(values, precision, delim=","):
	"""
	converts list of any type to delim separated string
	"""
	sValues = list(map(lambda v: toStr(v, precision), values))
	return delim.join(sValues)

def toIntList(values):
	"""
	convert to int list
	"""
	return list(map(lambda va: int(va), values))
		
def toFloatList(values):
	"""
	convert to float list
	"""
	return list(map(lambda va: float(va), values))

def toStrList(values, precision):
	"""
	convert to string list
	"""
	return list(map(lambda va: toStr(va, precision), values))

def typedValue(val):
	"""
	return typed value given string
	"""
	tVal = None
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
	"""
	filePaths = []
	for (thisDir, subDirs, fileNames) in os.walk(dirPath):
		for fileName in fileNames:
			filePaths.append(os.path.join(thisDir, fileName))
	filePaths.sort()
	return filePaths

def getFileContent(path, verbose):
	"""
	get file contents in directory
	"""
	# dcument list
	docComplete  = []
	filePaths = getAllFiles(path)
	filePaths

	# read files
	for filePath in filePaths:
		if verbose:
			print("next file " + filePath)
		with open(filePath, 'r') as contentFile:
			content = contentFile.read()
			docComplete.append(content)
	return (docComplete, filePaths)

def getFileLines(dirPath):
	"""
	get lines from a file
	"""
	lines = list()
	for li in fileRecGen(dirPath):
		lines.append(li)		
	return lines

def getFileColumnAsString(dirPath, index, delim=","):
	"""
	get string fileds from a file
	"""
	fields = list()
	for rec in fileRecGen(dirPath, delim):
		fields.append(rec[index])	
	#print(fields)	
	return fields

def getFileColumnAsFloat(dirPath, index, delim=","):
	"""
	get float fileds from a file
	"""
	#print("{}  {}".format(dirPath, index))
	fields = getFileColumnAsString(dirPath, index, delim=",")
	return list(map(lambda v:float(v), fields))
	
def getFileColumnAsInt(dirPath, index, delim=","):
	"""
	get float fileds from a file
	"""
	fields = getFileColumnAsString(dirPath, delim, index)
	return list(map(lambda v:int(v), fields))

def getFileAsIntMatrix(dirPath, columns, delim=","):
	"""
	extracts int matrix from csv file given column indices with each row being  concatenation of 
	extracted column values row size = num of columns
	"""
	mat = list()
	for rec in  fileSelFieldsRecGen(dirPath, columns, delim):
		mat.append(asIntList(rec))
	return mat

def getFileAsFloatMatrix(dirPath, columns, delim=","):
	"""
	extracts float matrix from csv file given column indices with each row being concatenation of  
	extracted column values row size = num of columns
	"""
	mat = list()
	for rec in  fileSelFieldsRecGen(dirPath, columns, delim):
		mat.append(asFloatList(rec))
	return mat
		
def getMultipleFileAsInttMatrix(dirPathWithCol,  delim=","):
	"""
	extracts float matrix from from csv files given column index for each file. 
	num of columns  = number of rows in each file and num of rows = number of files
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

def takeFirst(elem):
	"""
	sorting
	"""
	return elem[0]

def takeSecond(elem):
	"""
	sorting
	"""
	return elem[1]

def takeThird(elem):
	"""
	sorting
	"""
	return elem[2]

def addToKeyedCounter(dCounter, key, count):
	"""
	keyed counter
	"""
	curCount = dCounter.get(key, 0)
	dCounter[key] = curCount + count

def incrKeyedCounter(dCounter, key):
	"""
	keyed counter
	"""
	addToKeyedCounter(dCounter, key, 1)

def appendKeyedList(dList, key, elem):
	"""
	keyed list
	"""
	curList = dList.get(key, [])
	curList.append(elem)
	dList[key] = curList

def isNumber(st):
	"""
	Returns True is string is a number
	"""
	return st.replace('.','',1).isdigit()

def fileRecGen(filePath, delim = None):
	"""
	file record generator
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
	"""
	columns = strToIntArray(columns, delim)
	for rec in fileRecGen(dirPath, delim):
		extracted = extractList(rec, columns)
		yield extracted

def asIntList(items):
	"""
	returns int list
	"""
	return [int(i) for i in items]
			
def asFloatList(items):
	"""
	returns float list
	"""
	return [float(i) for i in items]

def pastTime(interval, unit):
	"""
	current and past time
	"""
	curTime = int(time.time())
	if unit == "d":
		pastTime = curTime - interval * secInDay
	elif unit == "h":
		pastTime = curTime - interval * secInHour
	elif unit == "m":
		pastTime = curTime - interval * secInMinute
	else:
		raise ValueError("invalid time unit")
	return (curTime, pastTime)

def minuteAlign(ts):
	"""
	minute aligned time	
	"""
	return int((ts / secInMinute)) * secInMinute

def multMinuteAlign(ts, min):
	"""
	multi minute aligned time	
	"""
	intv = secInMinute * min
	return int((ts / intv)) * intv

def hourAlign(ts):
	"""
	hour aligned time	
	"""
	return int((ts / secInHour)) * secInHour
	
def dayAlign(ts):
	"""
	day aligned time	
	"""
	return int((ts / secInDay)) * secInDay

def timeAlign(ts, unit):
	"""
	boundary alignment of time
	"""
	alignedTs = 0
	if unit == "m":
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
	"""
	rem = ts % secInYear
	dow = int(rem / secInMonth)
	return dow
		
def dayOfWeek(ts):
	"""
	day of week
	"""
	rem = ts % secInWeek
	dow = int(rem / secInDay)
	return dow

def hourOfDay(ts):
	"""
	hour of day
	"""
	rem = ts % secInDay
	hod = int(rem / secInHour)
	return hod
	
def processCmdLineArgs(expectedTypes, usage):
	"""
	process command line args and returns args as typed values
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
	"""
	mutations = set()
	for i in range(numMutate):
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
	return val

def swap(values, first, second):
	"""
	swap two elements
	"""
	t = values[first]
	values[first] = values[second]	
	values[second] = t

def createLogger(name, logFilePath, logLevName):
	"""
	creates logger
	"""
	logger = logging.getLogger(name)
	fHandler = logging.FileHandler(logFilePath)
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
	"""
	with open(os.devnull, "w") as devnull:
		oldStdout = sys.stdout
		sys.stdout = devnull
		try:  
			yield
		finally:
			sys.stdout = oldStdout
		
class StepFunction:
	"""
	step function
	"""
	def __init__(self,  *values):
		self.points = values
	
	def find(self, x):
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
	def __init__(self,  rowSize, catValues, trueVal, falseVal, delim):
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
		#print (row)
		rowArr = row.split(self.delim)
		assert len(rowArr) == self.rowSize, "row does not have expected number of columns " + str(len(rowArr))
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
		assert len(newRowArr) == self.newRowSize, "invalid new row size " + str(len(newRowArr))
		return self.delim.join(newRowArr)
		
		
