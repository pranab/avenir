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
		
		Parameters
			numIter :num of iterations
			callback : call back method
			logFilePath : log file path
			logLevName : log level
		"""
		self.samplers = list()
		self.numIter = numIter;
		self.callback = callback
		self.extraArgs = None
		self.output = list()
		self.sum = None
		self.mean = None
		self.sd = None
		self.replSamplers = dict()
		self.prSamples = None
		
		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(__name__, logFilePath, logLevName)
			self.logger.info("******** stating new  session of MonteCarloSimulator")


	def registerBernoulliTrialSampler(self, pr):
		"""
		bernoulli trial sampler
		
		Parameters
			pr : probability
		"""
		self.samplers.append(BernoulliTrialSampler(pr))
		
	def registerPoissonSampler(self, rateOccur, maxSamp):
		"""
		poisson sampler
		
		Parameters
			rateOccur : rate of occurence
			maxSamp : max limit on no of samples
		"""
		self.samplers.append(PoissonSampler(rateOccur, maxSamp))

	def registerUniformSampler(self, minv, maxv):
		"""
		uniform sampler
		
		Parameters
			minv : min value
			maxv : max value
		"""
		self.samplers.append(UniformNumericSampler(minv, maxv))

	def registerTriangularSampler(self, min, max, vertexValue, vertexPos=None):
		"""
		triangular sampler
		
		Parameters
			xmin : min  value
			xmax : max  value
			vertexValue : distr value at vertex
			vertexPos : vertex pposition
		"""
		self.samplers.append(TriangularRejectSampler(min, max, vertexValue, vertexPos))

	def registerGaussianSampler(self, mean, sd):
		"""
		gaussian sampler

		Parameters
			mean : mean
			sd : std deviation
		"""
		self.samplers.append(GaussianRejectSampler(mean, sd))
		
	def registerNormalSampler(self, mean, sd):
		"""
		gaussian sampler using numpy

		Parameters
			mean : mean
			sd : std deviation
		"""
		self.samplers.append(NormalSampler(mean, sd))

	def registerLogNormalSampler(self, mean, sd):
		"""
		log normal sampler using numpy

		Parameters
			mean : mean
			sd : std deviation
		"""
		self.samplers.append(LogNormalSampler(mean, sd))

	def registerParetoSampler(self, mode, shape):
		"""
		pareto sampler using numpy

		Parameters
			mode : mode
			shape : shape
		"""
		self.samplers.append(ParetoSampler(mode, shape))

	def registerGammaSampler(self, shape, scale):
		"""
		gamma sampler using numpy

		Parameters
			shape : shape
			scale : scale
		"""
		self.samplers.append(GammaSampler(shape, scale))

	def registerDiscreteRejectSampler(self, xmin, xmax, step, *values):
		"""
		disccrete int sampler

		Parameters
			xmin : min  value
			xmax : max  value
			step : discrete step
			values : distr values
		"""
		self.samplers.append(DiscreteRejectSampler(xmin, xmax, step, *values))

	def registerNonParametricSampler(self, minv, binWidth, *values):
		"""
		nonparametric sampler

		Parameters
			xmin : min  value
			binWidth : bin width
			values : distr values
		"""
		sampler = NonParamRejectSampler(minv, binWidth, *values)
		sampler.sampleAsFloat()
		self.samplers.append(sampler)

	def registerMultiVarNormalSampler(self,  numVar, *values):
		"""
		multi var gaussian sampler using numpy

		Parameters
			numVar : no of variables
			values : numVar mean values followed by numVar x numVar values for covar matrix
		"""
		self.samplers.append(MultiVarNormalSampler(numVar, *values))
		
	def registerJointNonParamRejectSampler(self, xmin, xbinWidth, xnbin, ymin, ybinWidth, ynbin, *values):
		"""
		joint nonparametric sampler

		Parameters
			xmin : min  value for x
			xbinWidth : bin width for x
			xnbin : no of bins for x
			ymin : min  value for y
			ybinWidth : bin width for y
			ynbin : no of bins for y
			values : distr values
		"""
		self.samplers.append(JointNonParamRejectSampler(xmin, xbinWidth, xnbin, ymin, ybinWidth, ynbin, *values))

	def registerRangePermutationSampler(self, minv, maxv, *numShuffles):
		"""
		permutation sampler with range

		Parameters
			minv : min of range
			maxv : max of range
			numShuffles : no of shuffles or range of no of shuffles
		"""
		self.samplers.append(PermutationSampler.createSamplerWithRange(minv, maxv, *numShuffles))
	
	def registerValuesPermutationSampler(self, values, *numShuffles):
		"""
		permutation sampler with values

		Parameters
			values : list data
			numShuffles : no of shuffles or range of no of shuffles
		"""
		self.samplers.append(PermutationSampler.createSamplerWithValues(values, *numShuffles))
	
	def registerNormalSamplerWithTrendCycle(self, mean, stdDev, trend, cycle,  step=1):
		"""
		normal sampler with trend and cycle

		Parameters
			mean : mean
			stdDev : std deviation
			dmean : trend delta
			cycle : cycle values wrt base mean
			step : adjustment step for cycle and trend
		"""
		self.samplers.append(NormalSamplerWithTrendCycle(mean, stdDev, trend, cycle,  step))

	def registerCustomSampler(self, sampler):
		"""
		eventsampler
		
		Parameters
			sampler : sampler with sample() method
		"""
		self.samplers.append(sampler)
	
	def registerEventSampler(self, intvSampler, valSampler=None):
		"""
		custom sampler
		
		Parameters
		Parameters
			intvSampler : interval sampler
			valSampler : value sampler
		"""
		self.samplers.append(EventSampler(intvSampler, valSampler))

	def setSampler(self, var, iter, sampler):
		"""
		set sampler for some variable when iteration reaches certain point

		Parameters
			var : sampler index
			iter : iteration count
			sampler : new sampler
		"""
		key = (var, iter)
		self.replSamplers[key] = sampler

	def registerExtraArgs(self, *args):
		"""
		extra args

		Parameters
			args : extra argument list
		"""
		self.extraArgs = args

	def replSampler(self, iter):
		"""
		replace samper for this iteration

		Parameters
			iter : iteration number
		"""
		if len(self.replSamplers) > 0:
			for v in range(self.numVars):
				key = (v, iter)
				if key in self.replSamplers:
					sampler = self.replSamplers[key]
					self.samplers[v] = sampler

	def run(self):
		"""
		run simulator
		"""
		self.sum = None
		self.mean = None
		self.sd = None
		self.numVars = len(self.samplers)
		vOut = 0

		#print(formatAny(self.numIter, "num iterations"))
		for i in range(self.numIter):
			self.replSampler(i)
			args = list()
			for s in self.samplers:
				arg = s.sample()
				if type(arg) is list:
					args.extend(arg)
				else:
					args.append(arg)
					
			slen = len(args)
			if self.extraArgs:
				args.extend(self.extraArgs)
			args.append(self)
			args.append(i)
			vOut = self.callback(args)	
			self.output.append(vOut)
			self.prSamples = args[:slen]
	
	def getOutput(self):
		"""
		get raw output
		"""
		return self.output
	
	def setOutput(self, values):
		"""
		set raw output

		Parameters
			values : output values
		"""
		self.output = values
		self.numIter = len(values)

	def drawHist(self, myTitle, myXlabel, myYlabel):
		"""
		draw histogram

		Parameters
			myTitle : title
			myXlabel : label for x
			myYlabel : label for y
		"""
		pyplot.hist(self.output, density=True)
		pyplot.title(myTitle)
		pyplot.xlabel(myXlabel)
		pyplot.ylabel(myYlabel)
		pyplot.show()	
		
	def getSum(self):
		"""
		get sum
		"""
		if not self.sum:
			self.sum = sum(self.output)
		return self.sum
		
	def getMean(self):
		"""
		get average
		"""
		if self.mean is None:
			self.mean = statistics.mean(self.output)
		return self.mean 
		
	def getStdDev(self):
		"""
		get std dev
		"""
		if self.sd is None:
			self.sd = statistics.stdev(self.output, xbar=self.mean) if self.mean else statistics.stdev(self.output)
		return self.sd 
		

	def getMedian(self):
		"""
		get average
		"""
		med = statistics.median(self.output)
		return med

	def getMax(self):
		"""
		get max
		"""
		return max(self.output)
		
	def getMin(self):
		"""
		get min
		"""
		return min(self.output)
		
	def getIntegral(self, bounds):
		"""
		integral

		Parameters
			bounds :  bound on sum
		"""
		if not self.sum:
			self.sum = sum(self.output)
		return self.sum * bounds / self.numIter
	
	def getLowerTailStat(self, zvalue, numIntPoints=50):
		"""
		get lower tail stat

		Parameters
			zvalue : zscore upper bound 
			numIntPoints : no of interpolation point for cum distribution
		"""
		mean = self.getMean()
		sd = self.getStdDev()
		tailStart = self.getMin()
		tailEnd = mean - zvalue * sd
		cvaCounts = self.cumDistr(tailStart, tailEnd, numIntPoints)
		
		reqConf = floatRange(0.0, 0.150, .01)	
		msg = "p value outside interpolation range, reduce zvalue and try again {:.5f}  {:.5f}".format(reqConf[-1], cvaCounts[-1][1])
		assert reqConf[-1] < cvaCounts[-1][1], msg
		critValues = self.interpolateCritValues(reqConf, cvaCounts, True, tailStart, tailEnd)
		return critValues
		
	def getPercentile(self, cvalue):
		"""
		percentile

		Parameters
			cvalue : value for percentile 
		"""
		count = 0
		for v in self.output:
			if v < cvalue:
				count += 1 
		percent =  int(count * 100.0 / self.numIter)
		return percent


	def getCritValue(self, pvalue):	
		"""
		critical value for probabaility threshold

		Parameters
			pvalue : pvalue 
		"""
		assertWithinRange(pvalue, 0.0, 1.0, "invalid probabaility value")
		svalues = self.output.sorted()
		ppval = None
		cpval = None
		intv = 1.0 / (self.numIter - 1)
		for i in range(self.numIter - 1):
			cpval = (i + 1) / self.numIter
			if cpval > pvalue:
				sl = svalues[i] - svalues[i-1]
				cval = svalues[i-1] + sl * (pvalue - ppval)
				break
			ppval = cpval
		return cval
		
		
	def getUpperTailStat(self, zvalue, numIntPoints=50):
		"""
		upper tail stat

		Parameters
			zvalue : zscore upper bound 
			numIntPoints : no of interpolation point for cum distribution
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
		
		Parameters
			tailStart : tail start
			tailEnd : tail end
			numIntPoints : no of interpolation points
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
			if self.logger is not None:
				self.logger.info("{:.3f}  {:.3f}".format(p[0], p[1]))
			cvaCounts.append(p)
		return cvaCounts
			
	def interpolateCritValues(self, reqConf, cvaCounts, lowertTail, tailStart, tailEnd):	
		"""
		interpolate for spefici confidence limits
		
		Parameters
			reqConf : confidence level values
			cvaCounts : cum values
			lowertTail : True if lower tail
			tailStart ; tail start
			tailEnd : tail end
		"""
		critValues = list()
		if self.logger is not None:
			self.logger.info("target conf limit " + str(reqConf))
		reqConfSub = reqConf[1:] if lowertTail else reqConf[:-1]
		for rc in reqConfSub:
			for i in range(len(cvaCounts) -1):
				if rc >= cvaCounts[i][1] and rc < cvaCounts[i+1][1]:
					#print("interpoltate between " + str(cvaCounts[i])  +  " and " + str(cvaCounts[i+1]))
					slope = (cvaCounts[i+1][0] - cvaCounts[i][0]) / (cvaCounts[i+1][1] - cvaCounts[i][1])
					cval = cvaCounts[i][0] + slope * (rc - cvaCounts[i][1]) 
					p = (rc, cval)
					if self.logger is not None:
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
