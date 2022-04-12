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

import sys
import random 
import time
import math
import numpy as np
import statistics 
from util import *

"""
histogram class
"""
class Histogram:
	def __init__(self, min, binWidth):
		"""
    	initializer
    	
		Parameters
			min : min x
			binWidth : bin width
    	"""
		self.xmin = min
		self.binWidth = binWidth
	
	@classmethod
	def createInitialized(cls, min, binWidth, values):
		"""
    	create histogram instance
    	
		Parameters
			min : min x
			binWidth : bin width
			values : y values
    	"""
		instance = cls(min, binWidth)
		instance.xmax = min + binWidth * (len(values) - 1)
		instance.ymin = 0
		instance.bins = np.array(values)
		instance.fmax = 0
		for v in values:
			if (v > instance.fmax):
				instance.fmax = v
		instance.ymin = 0.0
		instance.ymax = instance.fmax
		return instance
	
	@classmethod
	def createUninitialized(cls, min, max, binWidth):
		"""
    	create histogram instance with no y values
    	
		Parameters
			min : min x
			max : max x
			binWidth : bin width
    	"""
		instance = cls(min, binWidth)
		instance.xmax = max
		instance.numBin = (max - min) / binWidth + 1
		instance.bins = np.zeros(instance.numBin)
		return instance
	
	def initialize(self):
		"""
    	set y values to 0
    	"""
		self.bins = np.zeros(self.numBin)
		
	def add(self, value):
		"""
    	adds a value to a bin
    	
		Parameters
			value : value
    	"""
		bin = (value - self.xmin) / self.binWidth
		if (bin < 0 or  bin > self.numBin - 1):
			print (bin)
			raise ValueError("outside histogram range")
		self.bins[bin] += 1.0
	
	def normalize(self):
		"""
    	normalize  bin counts
    	"""
		total = self.bins.sum()
		self.bins = np.divide(self.bins, total)
	
	
	def cumDistr(self):
		"""
    	cumulative dists
    	"""
		self.cbins = np.cumsum(self.bins)
	
	def percentile(self, percent):
		"""
    	return value corresponding to a percentile
    	
		Parameters
			percent : percentile value
    	"""
		if self.cbins is None:
			raise ValueError("cumulative distribution is not available")
			
		for i,cuml in enumerate(self.cbins):
			if percent > cuml:
				value = (i * self.binWidth) - (self.binWidth / 2) + \
				(percent - self.cbins[i-1]) * self.binWidth / (self.cbins[i] - self.cbins[i-1]) 
				break
		return value
		
	def max(self):
		"""
    	return max bin value 
    	"""
		return self.bins.max()
	
	def value(self, x):
		"""
    	return a bin value	
     	
		Parameters
			x : x value
   		"""
		bin = int((x - self.xmin) / self.binWidth)
		f = self.bins[bin]
		return f
	
	def cumValue(self, x):
		"""
    	return a cumulative bin value	
     	
		Parameters
			x : x value
   		"""
		bin = int((x - self.xmin) / self.binWidth)
		c = self.cbins[bin]
		return c
	
		
	def getMinMax(self):
		"""
    	returns x min and x max
    	"""
		return (self.xmin, self.xmax)
		
	def boundedValue(self, x):
		"""
    	return x bounde by min and max	
     	
		Parameters
			x : x value
   		"""
		if x < self.xmin:
			x = self.xmin
		elif x > self.xmax:
			x = self.xmax
		return x

"""
categorical histogram class
"""
class CatHistogram:
	def __init__(self):
		"""
    	initializer
    	"""
		self.binCounts = dict()
		self.counts = 0
		self.normalized = False
	
	def add(self, value):
		"""
		adds a value to a bin
		
		Parameters
			x : x value
		"""
		addToKeyedCounter(self.binCounts, value)
		self.counts += 1	
		
	def normalize(self):
		"""
		normalize
		"""
		if not self.normalized:
			self.binCounts = dict(map(lambda r : (r[0],r[1] / self.counts), self.binCounts.items()))
			self.normalized = True
	
	def getMode(self):
		"""
		get mode
		"""
		maxk = None
		maxv = 0
		#print(self.binCounts)
		for  k,v  in  self.binCounts.items():
			if v > maxv:
				maxk = k
				maxv = v
		return (maxk, maxv)	
	
	def getEntropy(self):
		"""
		get entropy
		"""
		self.normalize()
		entr = 0 
		#print(self.binCounts)
		for  k,v  in  self.binCounts.items():
			entr -= v * math.log(v)
		return entr

	def getUniqueValues(self):
		"""
		get unique values
		"""		
		return list(self.binCounts.keys())

	def getDistr(self):
		"""
		get distribution
		"""	
		self.normalize()	
		return self.binCounts.copy()
		
class RunningStat:
	"""
	running stat class
	"""
	def __init__(self):
   		"""
    	initializer	
   		"""
   		self.sum = 0.0
   		self.sumSq = 0.0
   		self.count = 0
	
	@staticmethod
	def create(count, sum, sumSq):
		"""
    	creates iinstance	
     	
		Parameters
			sum : sum of values
			sumSq : sum of valure squared
		"""
		rs = RunningStat()
		rs.sum = sum
		rs.sumSq = sumSq
		rs.count = count
		return rs
		
	def add(self, value):
		"""
		adds new value

		Parameters
			value : value to add
		"""
		self.sum += value
		self.sumSq += (value * value)
		self.count += 1

	def getStat(self):
		"""
		return mean and std deviation 
		"""
		mean = self.sum /self. count
		t = self.sumSq / (self.count - 1) - mean * mean * self.count / (self.count - 1)
		sd = math.sqrt(t)
		re = (mean, sd)
		return re

	def addGetStat(self,value):
		"""
		calculate mean and std deviation with new value added

		Parameters
			value : value to add
		"""
		self.add(value)
		re = self.getStat()
		return re
	
	def getCount(self):
		"""
		return count
		"""
		return self.count
	
	def getState(self):
		"""
		return state
		"""
		s = (self.count, self.sum, self.sumSq)
		return s
		
class SlidingWindowStat:
	"""
	sliding window stats
	"""
	def __init__(self):
		"""
		initializer
		"""
		self.sum = 0.0
		self.sumSq = 0.0
		self.count = 0
		self.values = None
	
	@staticmethod
	def create(values, sum, sumSq):
		"""
    	creates iinstance	
     	
		Parameters
			sum : sum of values
			sumSq : sum of valure squared
		"""
		sws = SlidingWindowStat()
		sws.sum = sum
		sws.sumSq = sumSq
		self.values = values.copy()
		sws.count = len(self.values)
		return sws
		
	@staticmethod
	def initialize(values):
		"""
    	creates iinstance	
     	
		Parameters
			values : list of values
		"""
		sws = SlidingWindowStat()
		sws.values = values.copy()
		for v in sws.values:
			sws.sum += v
			sws.sumSq += v * v		
		sws.count = len(sws.values)
		return sws

	@staticmethod
	def createEmpty(count):
		"""
    	creates iinstance	
     	
		Parameters
			count : count of values
		"""
		sws = SlidingWindowStat()
		sws.count = count
		sws.values = list()
		return sws

	def add(self, value):
		"""
		adds new value
		
		Parameters
			value : value to add
		"""
		self.values.append(value)		
		if len(self.values) > self.count:
			self.sum += value - self.values[0]
			self.sumSq += (value * value) - (self.values[0] * self.values[0])
			self.values.pop(0)
		else:
			self.sum += value
			self.sumSq += (value * value)
		

	def getStat(self):
		"""
		calculate mean and std deviation 
		"""
		mean = self.sum /self. count
		t = self.sumSq / (self.count - 1) - mean * mean * self.count / (self.count - 1)
		sd = math.sqrt(t)
		re = (mean, sd)
		return re

	def addGetStat(self,value):
		"""
		calculate mean and std deviation with new value added
		"""
		self.add(value)
		re = self.getStat()
		return re
	
	def getCount(self):
		"""
		return count
		"""
		return self.count
	
	def getCurSize(self):
		"""
		return count
		"""
		return len(self.values)
		
	def getState(self):
		"""
		return state
		"""
		s = (self.count, self.sum, self.sumSq)
		return s
		

def basicStat(ldata):
	"""
	mean and std dev

	Parameters
		ldata : list of values
	"""
	m = statistics.mean(ldata)
	s = statistics.stdev(ldata, xbar=m)
	r = (m, s)
	return r

def getFileColumnStat(filePath, col, delem=","):
	"""
	gets stats for a file column
	
	Parameters
		filePath : file path
		col : col index
		delem : field delemter
	"""
	rs = RunningStat()
	for rec in fileRecGen(filePath, delem):
		va = float(rec[col])
		rs.add(va)
		
	return rs.getStat()