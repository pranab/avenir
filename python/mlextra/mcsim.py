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
from matplotlib import pyplot
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

class MonteCarloSimulator(object):
	"""
	monte carlo simulator for intergation, various statistic for complex fumctions
	"""
	def __init__(self, numIter, callback, logFilePath, logLevName):
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
		self.sd = None

		self.logger = createLogger(__name__, logFilePath, logLevName)
		self.logger.info("******** stating new  session of MonteCarloSimulator")

	def registerBernoulliTrialSampler(self, pr):
		"""
		bernoulli trial sampler
		"""
		self.samplers.append(BernoulliTrialSampler(pr))
		
	def registerPoissonSampler(self, rateOccur, maxSamp):
		"""
		poisson sampler
		"""
		self.samplers.append(PoissonSampler(rateOccur, maxSamp))

	def registerUniformSampler(self, min, max):
		"""
		uniform sampler
		"""
		self.samplers.append(UniformNumericSampler(min, max))

	def registerTriangularSampler(self, min, max, vertexValue, vertexPos=None):
		"""
		triangular sampler
		"""
		self.samplers.append(TriangularRejectSampler(min, max, vertexValue, vertexPos))

	def registerGaussianSampler(self, mean, sd):
		"""
		gaussian sampler
		"""
		self.samplers.append(GaussianRejectSampler(mean, sd))
		
	def registerNormalSampler(self, mean, sd):
		"""
		gaussian sampler using numpy
		"""
		self.samplers.append(NormalSampler(mean, sd))

	def registerLogNormalSampler(self, mean, sd):
		"""
		log normal sampler using numpy
		"""
		self.samplers.append(LogNormalSampler(mean, sd))

	def registerParetoSampler(self, mode, shape):
		"""
		pareto sampler using numpy
		"""
		self.samplers.append(ParetoSampler(mode, shape))

	def registerGammaSampler(self, shape, scale):
		"""
		gamma sampler using numpy
		"""
		self.samplers.append(GammaSampler(shape, scale))

	def registerDiscreteRejectSampler(self, xmin, xmax, step, *values):
		"""
		disccrete int sampler
		"""
		self.samplers.append(DiscreteRejectSampler(xmin, xmax, step, *values))

	def registerNonParametricSampler(self, min, binWidth, *values):
		"""
		nonparametric sampler
		"""
		sampler = NonParamRejectSampler(min, binWidth, *values)
		sampler.sampleAsFloat()
		self.samplers.append(sampler)

	def registerMultiVarNormalSampler(self,  numVar, *values):
		"""
		multi var gaussian sampler using numpy
		"""
		self.samplers.append(MultiVarNormalSampler(numVar, *values))
		
	def registerJointNonParamRejectSampler(self, xmin, xbinWidth, xnbin, ymin, ybinWidth, ynbin, *values):
		"""
		joint nonparametric sampler
		"""
		self.samplers.append(JointNonParamRejectSampler(xmin, xbinWidth, xnbin, ymin, ybinWidth, ynbin, *values))

	def registerRangePermutationSampler(self, min, max, *numShuffles):
		"""
		permutation sampler with range
		"""
		self.samplers.append(PermutationSampler.createSamplerWithRange(min, max, *numShuffles))
	
	def registerValuesPermutationSampler(self, values):
		"""
		permutation sampler with values
		"""
		self.samplers.append(PermutationSampler.createSamplerWithValues(values, *numShuffles))
	
	def registerCustomSampler(self, sampler):
		"""
		permutation sampler with values
		"""
		self.samplers.append(sampler)
	
	def registerExtraArgs(self, *args):
		"""
		extra args
		"""
		self.extraArgs = args

	def run(self):
		"""
		run simulator
		"""
		self.sum = None
		self.mean = None
		self.sd = None

		#print(formatAny(self.numIter, "num iterations"))
		for i in range(self.numIter):
			args = list()
			for s in self.samplers:
				arg = s.sample()
				if type(arg) is list:
					args.extend(arg)
				else:
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
	
	def drawHist(self, myTitle, myXlabel, myYlabel):
		"""
		draw histogram
		"""
		pyplot.hist(self.output, density=True)
		pyplot.title(myTitle)
		pyplot.xlabel(myXlabel)
		pyplot.ylabel(myYlabel)
		pyplot.show()	
		
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
		if self.mean is None:
			self.mean = statistics.mean(self.output)
		return self.mean 
		
	def getStdDev(self):
		"""
		std dev
		"""
		if self.sd is None:
			self.sd = statistics.stdev(self.output, xbar=self.mean) if self.mean else statistics.stdev(self.output)
		return self.sd 
		

	def getMedian(self):
		"""
		average
		"""
		med = statistics.median(self.output)
		return med

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
		
		reqConf = floatRange(0.0, 0.158, .01)	
		msg = "p value outside interpolation range, reduce zvalue and try again {:.5f}  {:.5f}".format(reqConf[-1], cvaCounts[-1][1])
		assert reqConf[-1] < cvaCounts[-1][1], msg
		critValues = self.interpolateCritValues(reqConf, cvaCounts, True, tailStart, tailEnd)
		return critValues
		
	def getPercentile(self, value):
		"""
		percentile
		"""
		count = 0
		for v in self.output:
			if v < value:
				count += 1 
		percent =  int(count * 100.0 / self.numIter)
		return percent
		
	def getUpperTailStat(self, zvalue, numIntPoints=50):
		"""
		upper tail stat
		"""
		mean = self.getMean()
		sd = self.getStdDev()
		tailStart = mean + zvalue * sd
		tailEnd = self.getMax()
		cvaCounts = self.cumDistr(tailStart, tailEnd, numIntPoints)		
		
		reqConf = floatRange(0.85, 1.0, .01)	
		msg = "p value outside interpolation range, reduce zvalue and try again {:.5f}  {:.5f}".format(reqConf[0], cvaCounts[0][1])
		assert reqConf[0] > cvaCounts[0][1],  msg
		critValues = self.interpolateCritValues(reqConf, cvaCounts, False, tailStart, tailEnd)
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
			self.logger.info("{:.3f}  {:.3f}".format(p[0], p[1]))
			cvaCounts.append(p)
		return cvaCounts
			
	def interpolateCritValues(self, reqConf, cvaCounts, lowertTail, tailStart, tailEnd):	
		"""
		interpolate for spefici confidence limits
		"""
		critValues = list()
		self.logger.info("target conf limit " + str(reqConf))
		reqConfSub = reqConf[1:] if lowertTail else reqConf[:-1]
		for rc in reqConfSub:
			for i in range(len(cvaCounts) -1):
				if rc >= cvaCounts[i][1] and rc < cvaCounts[i+1][1]:
					#print("interpoltate between " + str(cvaCounts[i])  +  " and " + str(cvaCounts[i+1]))
					slope = (cvaCounts[i+1][0] - cvaCounts[i][0]) / (cvaCounts[i+1][1] - cvaCounts[i][1])
					cval = cvaCounts[i][0] + slope * (rc - cvaCounts[i][1]) 
					p = (rc, cval)
					self.logger.debug("interpolated crit values {:.3f} {:.3f}".format(p[0], p[1]))
					critValues.append(p)
					break
		if lowertTail:
			p = (0.0, tailStart)
			critValues.insert(0, p)
		else:
			p = (1.0, tailEnd)
			critValues.append(p)
		return critValues
