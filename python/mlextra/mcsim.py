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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import random
import jprops
import statistics 
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

class MonteCarloSimulator(object):
	"""
	monte carlo simulator for intergation, various statistic for complex fumctions
	"""
	def __init__(self, numIter, callback):
		"""
		constructor
		"""
		self.samplers = list()
		self.numIter = numIter;
		self.callback = callback
		self.extraArgs = None
		self.output = list()
		self.sum = None
		self.mean = None
		
	def registerIntUniforSampler(self, min, max):
		"""
		int uniform sampler
		"""
		self.samplers.append(UniformNumericSampler(min, max))
		
	def registerFloatUniforSampler(self, min, max):
		"""
		float uniform sampler
		"""
		self.samplers.append(UniformNumericSampler(min, max))

	def registerIntGaussianSampler(self, mean, sd):
		"""
		int gaussian sampler
		"""
		sampler = GaussianRejectSampler(mean, sd)
		sampler.sampleAsInt()
		self.samplers.append(sampler)

	def registerFloatGaussianSampler(self, mean, sd):
		"""
		float gaussian sampler
		"""
		self.samplers.append(GaussianRejectSampler(mean, sd))
		
	def registerIntNonParametricSampler(self, min, binWidth, *values):
		"""
		int nonparametric sampler
		"""
		self.samplers.append(NonParamRejectSampler(min, binWidth, values))

	def registerFloatNonParametricSampler(self, min, binWidth, *values):
		"""
		int nonparametric sampler
		"""
		sampler = NonParamRejectSampler(min, binWidth, values)
		sampler.sampleAsFloat()
		self.samplers.append(sampler)
		
	def registerRangePermutationSampler(self, min, max):
		"""
		permutation sampler with range
		"""
		self.samplers.append(PermutationSampler.createSamplerWithRange(min, max))
	
	def registerValuesPermutationSampler(self, values):
		"""
		permutation sampler with values
		"""
		self.samplers.append(PermutationSampler.createSamplerWithValues(values))
	
	def registerExtraAfgs(self, args):
		"""
		extra args
		"""
		self.extraArgs = args

	def run(self):
		"""
		run simulator
		"""
		for i in range(self.numIter):
			args = list()
			for s in self.samplers:
				arg = s.samples()
				args.append(arg)
			if self.extraArgs:
				args.extend(self.extraArgs)
			vOut = self.callback(args)	
			self.output.append(vOut)
	
	def getOutput(self):
		"""
		raw output
		"""
		return self.output
		
	def getSum(self):
		"""
		sum
		"""
		if not self.sum:
			self.sum = sum(self.output)
		return self.sum
		
	def getMean(self):
		"""
		average
		"""
		self.mean = statistics.mean(self.output)
		return self.mean 
		
	def getStdDev(self):
		"""
		std dev
		"""
		return statistics.stdev(self.output, xbar=self.mean) if self.mean else statistics.stdev(self.output)

	def getMedian(self):
		"""
		average
		"""
		return statistics.median(self.output)

	def getMax(self):
		"""
		max
		"""
		return max(self.output)
		
	def getMin(self):
		"""
		min
		"""
		return min(self.output)
		
	def getIntegral(self, bounds):
		"""
		integral
		"""
		if not self.sum:
			self.sum = sum(self.output)
		return self.sum * bounds / self.numIter
	
	def getLowerTailStat(self, zvalue, numIntPoints=50):
		"""
		lower tail stat
		"""
		mean = self.getMean()
		sd = self.getStdDev()
		tailStart = self.getMin()
		tailEnd = mean - zvalue * sd
		cvaCounts = self.cumDistr(tailStart, tailEnd, numIntPoints)
		
		reqConf = floatRange(0.0, 0.1, .01)	
		assert reqConf[-1] > cvaCounts[-1][1],  "p value outside interpolation range, reduce zvalue and try again"
		critValues = self.interpolateCritValues(reqConf, cvaCounts, True, tailStart, tailEnd)
		return critValues
		

	def getUpperTailStat(self, zvalue, numIntPoints=50):
		"""
		upper tail stat
		"""
		mean = self.getMean()
		sd = self.getStdDev()
		tailStart = mean + zvalue * sd
		tailEnd = self.getMax()
		cvaCounts = self.cumDistr(tailStart, tailEnd, numIntPoints)		
		
		reqConf = floatRange(0.9, 1.0, .01)	
		assert reqConf[0] > cvaCounts[0][1],  "p value outside interpolation range, reduce zvalue and try again"
		critValues = self.interpolateCritValues(reqConf, cvaCounts, False, tailStart, tailEnd)
		return critValues		
		return critValues

	def cumDistr(self, tailStart, tailEnd, numIntPoints):
		"""
		cumulative distribution at tail
		"""
		delta = (tailEnd - tailStart) / numIntPoints
		cvalues = floatRange(tailStart, tailEnd, delta)
		cvaCounts = list()
		for cv in cvalues:
			count = 0
			for v in self.output:
				if v < cv:
					count += 1
			p = (cv, count/self.numIter)
			cvaCounts.append(p)
		return cvaCounts
			
	def interpolateCritValues(self, reqConf, cvaCounts, lowertTail, tailStart, tailEnd):	
		"""
		interpolate for spefici confidence limits
		"""
		critValues = list()
		reqConfSub = reqConf[1:] if lowertTail else reqConf[:-1]
		for rc in reqConfSub:
			for i in range(len(cvaCounts) -1):
				if rc >= cvaCounts[i][1] and rc < cvaCounts[i+1][1]:
					slope = (cvaCounts[i+1][0] - cvaCounts[i][0]) / (cvaCounts[i+1][1] - cvaCounts[i][1])
					cval = cvaCounts[i][0] + slope * (rc - cvaCounts[i][1]) 
					p = (rc, cval)
					critValues.append(p)
					break
		if lowertTail:
			critValues.insert(0.0, tailStart)
		else:
			critValues.append(1.0, tailEnd)
		return critValues
