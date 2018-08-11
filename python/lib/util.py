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

tokens = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M",
	"N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]
numTokens = tokens[:10]
alphaTokens = tokens[10:36]

#generates ID
def genID(len):
	id = ""
	for i in range(len):
		id = id + selectRandomFromList(tokens)
	return id

#generates ID consisting of digits only		
def genNumID(len):
	id = ""
	for i in range(len):
		id = id + selectRandomFromList(numTokens)
	return id

#select an element randomly from a list		
def selectRandomFromList(list):
	return list[randint(0, len(list)-1)]

#generates random sublist from a list	
def selectRandomSubListFromList(list, num):
	sel = selectRandomFromList(list)
	selSet = {sel}
	selList = [sel]
	while (len(selSet) < num):
		sel = selectRandomFromList(list)
		if (sel not in selSet):
			selSet.add(sel)
			selList.append(sel)		
	return selList

#generates IP address	
def genIpAddress():
	i1 = randint(0,256)
	i2 = randint(0,256)
	i3 = randint(0,256)
	i4 = randint(0,256)
	ip = "%d.%d.%d.%d" %(i1,i2,i3,i4)
	return ip

#current time in ms	
def curTimeMs():
	return int((datetime.utcnow() - datetime(1970,1,1)).total_seconds() * 1000)

#second deg polynomial 	
def secDegPolyFit(x1, y1, x2, y2, x3, y3):
	t = (y1 - y2) / (x1 - x2)
	a = t - (y2 - y3) / (x2 - x3)
	a = a / (x1 - x3)
	b = t - a * (x1 + x2)
	c = y1 - a * x1 * x1 - b * x1
	return (a, b, c)

#range limit
def range_limit(val, min, max):
	if (val < min):
		val = min
	elif (val > max):
		val = max
	return val	

#strips number of chars from both ends	
def stripFileLines(filePath, offset):
	fp = open(filePath, "r")
	for line in fp:
		stripped = line[offset:len(line) - 1 - offset]
		print stripped
	fp.close()

# generate lat log within limits
def genLatLong(lat1, long1, lat2, long2):
 	lat = lat1 + (lat2 - lat1) * random.random()
	long = long1 + (long2 - long1) * random.random()
	return (lat, long)

#min limit
def minLimit(val, limit):
	if (val < limit):
		val = limit
	return val;

# max limit
def maxLimit(val, limit):
	if (val > limit):
		val = limit
	return val;

# if out side range sample within range
def rangeSample(val, minLim, maxLim):
	if val < minLim or val > maxLim:
		val = randint(minLim, maxLim)
	return val

# breaks a line into fields and keeps only specified fileds and returns new line
def extractFields(line, delim, keepIndices):
	items = line.split(delim)
	newLine = []
	for i in keepIndices:
		newLine.append(line[i])
	return delim.join(newLine)

def remFields(line, delim, remIndices):
	items = line.split(delim)
	newLine = []
	for i in range(len(items)):
		if not arrayContains(remIndices, i):
			newLine.append(line[i])
	return delim.join(newLine)

# checks if array contains an item 	
def arrayContains(arr, item):
	contains = True
	try:
		arr.index(item)
	except ValueError:
		contains = False
	return contains

# int array from delim separated string
def strToIntArray(line, delim):	
	arr = line.split(delim)
	return [int(a) for a in arr]

# return typed value given string
def typedValue(val):
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
	
#get all files recursively
def getAllFiles(dirPath):
	filePaths = []
	for (thisDir, subDirs, fileNames) in os.walk(dirPath):
		for fileName in fileNames:
			filePaths.append(os.path.join(thisDir, fileName))
	return filePaths

# get file content
def getFileContent(path, verbose):
	# dcument list
	docComplete  = []
	filePaths = getAllFiles(path)

	# read files
	for filePath in filePaths:
		if verbose:
			print "next file " + filePath
		with open(filePath, 'r') as contentFile:
			content = contentFile.read()
			docComplete.append(content)
	return (docComplete, filePaths)

# soring
def takeFirst(elem):
    return elem[0]

# soring
def takeSecond(elem):
    return elem[1]

# soring
def takeThird(elem):
    return elem[2]

# keyed counter
def addToKeyedCounter(dCounter, key, count):
	curCount = dCounter.get(key, 0)
	dCounter[key] = curCount + count

# keyed counter
def incrKeyedCounter(dCounter, key):
	addToKeyedCounter(dCounter, key, 1)

# keyed list
def appendKeyedList(dList, key, elem):
	curList = dList.get(key, [])
	curList.append(elem)
	dList[key] = curList

# Returns True is string is a number
def isNumber(st):
    return st.replace('.','',1).isdigit()

# file record generator
def fileRecGen(filePath, delim = None):
	with open(filePath, "r") as fp:
		for line in fp:	
			line = line[:-1]
			if delim is not None:
				line = line.split(delim)
			yield line

# returns int list
def asIntList(items):
	return [int(i) for i in items]
			
# returns float list
def asFloatList(items):
	return [float(i) for i in items]

# mutate string multiple times
def mutateString(val, numMutate, ctype):
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
			
# step function
class StepFunction:
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
		
	

# dummy variable generator for categorical variables
class DummyVarGenerator:
	def __init__(self,  rowSize, catValues, trueVal, falseVal, delim):
		self.rowSize = rowSize
		self.catValues = catValues
		numCatVar = len(catValues)
		colCount = 0
		for v in self.catValues.values():
			colCount += len(v)
		self.newRowSize = rowSize - numCatVar + colCount
		#print "new row size %d" %(self.newRowSize)
		self.trueVal = trueVal
		self.falseVal = falseVal
		self.delim = delim
	
	def processRow(self, row):	
		rowArr = row.split(self.delim)
		assert len(rowArr) == self.rowSize, "row does not have expected number of columns" + str(len(rowArr))
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
		
		
