#!/Users/pranab/Tools/anaconda/bin/python

import sys
import random 
import time
import math
import numpy as np


# histogram class
class Histogram:
	def __init__(self, min, binWidth):
		self.xmin = min
		self.binWidth = binWidth
	
	# create with bins already created	
	@classmethod
	def createWithData(cls, min, binWidth, values):
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
	
	# create with un initialized bins
	@classmethod
	def createUninitialized(cls, min, max, binWidth):
		instance = cls(min, binWidth)
		instance.xmax = max
		numBin = (max - min) / binWidth + 1
		instance.bins = np.zeros(numBin)
		return instance
		
	# add a value to a bin	
	def add(self, value):
		bin = (value - self.xmin) / self.binWidth 
		self.bins[bin] += 1.0
	
	# normalize 	
	def normalize(self):
		total = self.bins.sum()
		self.bins = np.divide(self.bins, total)
	
	# return max bin value	
	def max(self):
		return self.bins.max()
	
	# return a bin value	
	def value(self, x):
		bin = int((x - self.xmin) / self.binWidth)
		f = self.bins[bin]
		return f
		
	def getMinMax(self):
		return (self.xmin, self.xmax)
		
	def boundedValue(self, x):
		if x < self.xmin:
			x = self.xmin
		elif x > self.xmax:
			x = self.xmax
		return x
