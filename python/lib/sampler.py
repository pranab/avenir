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
import random
import numpy as np
from scipy import stats
from random import randint
from util import *

from stats import Histogram

def randomFloat(low, high):
	"""
	sample float within range

	Parameters
		low : low valuee
		high : high valuee
	"""
	return random.random() * (high-low) + low

def randomInt(minv, maxv):
	"""
	sample int within range

	Parameters
		minv : low valuee
		maxv : high valuee
	"""
	return randint(minv, maxv)
	
def randIndex(lData):
	"""
	random index of a list

	Parameters
		lData : list data
	"""
	return randint(0, len(lData)-1)

def randomUniformSampled(low, high):
	"""
	sample float within range
	
	Parameters
		low : low value
		high : high value
	"""
	return np.random.uniform(low, high)

def randomUniformSampledList(low, high, size):
	"""
	sample floats within range to create list

	Parameters
		low : low value
		high : high value
		size ; size of list to be returned
	"""
	return np.random.uniform(low, high, size)

def randomNormSampled(mean, sd):
	"""
	sample float from normal

	Parameters
		mean : mean
		sd : std deviation
	"""
	return np.random.normal(mean, sd)
	
def randomNormSampledList(mean, sd, size):
	"""
	sample float list from normal 

	Parameters
		mean : mean
		sd : std deviation
		size : size of list to be returned
	"""
	return np.random.normal(mean, sd, size)

def randomSampledList(sampler, size):
	"""
	sample list from given sampler 

	Parameters
		sampler : sampler object
		size : size of list to be returned
	"""
	return list(map(lambda i : sampler.sample(), range(size)))
	

def minLimit(val, minv):
	"""
	min limit
	
	Parameters
		val : value
		minv : min limit
	"""
	if (val < minv):
		val = minv
	return val

	
def rangeLimit(val, minv, maxv):
	"""
	range limit

	Parameters
		val : value
		minv : min limit
		maxv : max limit
	"""
	if (val < minv):
		val = minv
	elif (val > maxv):
		val = maxv
	return val


def sampleUniform(minv, maxv):
	"""
	sample int within range

	Parameters
		minv ; int min limit
		maxv : int max limit
	"""
	return randint(minv, maxv)


def sampleFromBase(value, dev):
	"""
	sample int wrt base

	Parameters
		value : base value
		dev : deviation
	"""
	return randint(value - dev, value + dev)


def sampleFloatFromBase(value, dev):
	"""
	sample float wrt base

	Parameters
		value : base value
		dev : deviation
	"""
	return randomFloat(value - dev, value + dev)


def distrUniformWithRanndom(total, numItems, noiseLevel):
	"""
	uniformly distribute with some randomness and preserves total

	Parameters
		total : total count
		numItems : no of bins
		noiseLevel : noise level fraction
	"""
	perItem = total / numItems
	var = perItem * noiseLevel
	items = []
	for i in range(numItems):
		item = perItem + randomFloat(-var, var)
		items.append(item)	
	
	#adjust last item
	sm = sum(items[:-1])
	items[-1] = total - sm
	return items


def isEventSampled(threshold, maxv=100):
	"""
	sample event which occurs if sampled below threshold

	Parameters
		threshold : threshold for sampling
		maxv : maximum values
	"""
	return randint(0, maxv) < threshold


def sampleBinaryEvents(events, probPercent):
	"""
	sample binary events

	Parameters
		events : two events
		probPercent : probability as percentage
	"""
	if (randint(0, 100) < probPercent):
		event = events[0]
	else:
		event = events[1]
	return event


def addNoiseNum(value, sampler):
	"""
	add noise to numeric value

	Parameters
		value : base value
		sampler : sampler for noise
	"""
	return value * (1 + sampler.sample())

	
def addNoiseCat(value, values, noise):	
	"""
	add noise to categorical value i.e with some probability change value

	Parameters
		value : cat value
		values : cat values
		noise : noise level fraction
	"""
	newValue = value
	threshold = int(noise * 100)
	if (isEventSampled(threshold)):		
		newValue = selectRandomFromList(values)
		while newValue == value:
			newValue = selectRandomFromList(values)
	return newValue


def sampleWithReplace(data, sampSize):
	"""
	sample with replacement

	Parameters
		data : array
		sampSize : sample size
	"""
	sampled = list()
	le = len(data)
	if sampSize is None:
		sampSize = le
	for i in range(sampSize):
		j = random.randint(0, le - 1)
		sampled.append(data[j])
	return sampled

class CumDistr:
	"""
	cumulative distr
	"""
	
	def __init__(self, data, numBins = None):
		"""
		initializer
		
		Parameters
			data : array
			numBins : no of bins
		"""
		if not numBins:
			numBins = int(len(data) / 5)
		res = stats.cumfreq(data, numbins=numBins)
		self.cdistr = res.cumcount / len(data)
		self.loLim = res.lowerlimit
		self.upLim = res.lowerlimit + res.binsize * res.cumcount.size
		self.binWidth = res.binsize
		
	def getDistr(self, value):
		"""
		get cumulative distribution
		
		Parameters
			value : value
		"""
		if value <= self.loLim:
			d = 0.0
		elif value >= self.upLim:
			d = 1.0
		else:
			bin = int((value - self.loLim) / self.binWidth)
			d = self.cdistr[bin]
		return d

class BernoulliTrialSampler:
	"""
	bernoulli trial sampler return True or False
	"""
	
	def __init__(self, pr):
		"""
		initializer
		
		Parameters
			pr : probability
		"""
		self.pr = pr
		
	def sample(self):
		"""
		samples value
		"""
		return random.random() < self.pr
	
class PoissonSampler:
	"""
	poisson sampler returns number of events
	"""
	def __init__(self, rateOccur, maxSamp):
		"""
		initializer
		
		Parameters
			rateOccur : rate of occurence
			maxSamp : max limit on no of samples
		"""
		self.rateOccur = rateOccur
		self.maxSamp = int(maxSamp)
		self.pmax = self.calculatePr(rateOccur)

	def calculatePr(self, numOccur):
		"""
		calulates probability
		
		Parameters
			numOccur : no of occurence
		"""
		p = (self.rateOccur ** numOccur) * math.exp(-self.rateOccur) / math.factorial(numOccur)
		return p

	def sample(self):
		"""
		samples value
		"""
		done = False
		samp = 0
		while not done:
			no = randint(0, self.maxSamp)
			sp = randomFloat(0.0, self.pmax)
			ap = self.calculatePr(no)
			if sp < ap:
				done = True
				samp = no
		return samp

class ExponentialSampler:
	"""
	returns interval between events
	"""
	def __init__(self, rateOccur, maxSamp = None):
		"""
		initializer
		
		Parameters
			rateOccur : rate of occurence
			maxSamp : max limit on interval
		"""
		self.interval = 1.0 / rateOccur
		self.maxSamp = int(maxSamp) if maxSamp is not None else None

	def sample(self):
		"""
		samples value
		"""
		sampled = np.random.exponential(scale=self.interval)
		if self.maxSamp is not None:
			while sampled > self.maxSamp:
				sampled = np.random.exponential(scale=self.interval)
		return sampled

class UniformNumericSampler:
	"""
	uniform sampler for numerical values
	"""
	def __init__(self, minv, maxv):
		"""
		initializer
		
		Parameters
			minv : min value
			maxv : max value
		"""
		self.minv = minv
		self.maxv = maxv

	def isNumeric(self):
		"""
		returns true
		"""
		return True
	
	def sample(self):
		"""
		samples value
		"""
		samp =	sampleUniform(self.minv, self.maxv) if isinstance(self.minv, int) else randomFloat(self.minv, self.maxv)
		return samp	

class UniformCategoricalSampler:
	"""
	uniform sampler for categorical values
	"""
	def __init__(self, cvalues):
		"""
		initializer
		
		Parameters
			cvalues : categorical value list
		"""
		self.cvalues = cvalues
	
	def isNumeric(self):
		return False

	def sample(self):
		"""
		samples value
		"""
		return selectRandomFromList(self.cvalues)	

class NormalSampler:
	"""
	normal sampler
	"""
	def __init__(self, mean, stdDev):
		"""
		initializer
		
		Parameters
			mean : mean
			stdDev : std deviation
		"""
		self.mean = mean
		self.stdDev = stdDev
		self.sampleAsInt = False

	def isNumeric(self):
		return True

	def sampleAsIntValue(self):
		"""
		set True to sample as int
		"""
		self.sampleAsInt = True
		
	def sample(self):
		"""
		samples value
		"""
		samp =  np.random.normal(self.mean, self.stdDev)
		if self.sampleAsInt:
			samp = int(samp)
		return samp
				
class LogNormalSampler:
	"""
	log normal sampler
	"""
	def __init__(self, mean, stdDev):
		"""
		initializer
		
		Parameters
			mean : mean
			stdDev : std deviation
		"""
		self.mean = mean
		self.stdDev = stdDev

	def isNumeric(self):
		return True

	def sample(self):
		"""
		samples value
		"""
		return np.random.lognormal(self.mean, self.stdDev)

class NormalSamplerWithTrendCycle:
	"""
	normal sampler with cycle and trend
	"""
	def __init__(self, mean, stdDev, dmean, cycle,  step=1):
		"""
		initializer
		
		Parameters
			mean : mean
			stdDev : std deviation
			dmean : trend delta
			cycle : cycle values wrt base mean
			step : adjustment step for cycle and trend
		"""
		self.mean = mean
		self.cmean = mean
		self.stdDev = stdDev
		self.dmean = dmean
		self.cycle = cycle
		self.clen = len(cycle) if cycle is not None else 0
		self.step = step
		self.count = 0

	def isNumeric(self):
		return True

	def sample(self):
		"""
		samples value
		"""
		s = np.random.normal(self.cmean, self.stdDev)
		self.count += 1
		if self.count % self.step == 0:
			cy = 0
			if self.clen > 1:
				coff =  self.count % self.clen
				cy = self.cycle[coff]
			tr = self.count * self.dmean
			self.cmean = self.mean + tr + cy
		return s


class ParetoSampler:
	"""
	pareto sampler
	"""
	def __init__(self, mode, shape):
		"""
		initializer
		
		Parameters
			mode : mode
			shape : shape
		"""
		self.mode = mode
		self.shape = shape

	def isNumeric(self):
		return True

	def sample(self):
		"""
		samples value
		"""
		return (np.random.pareto(self.shape) + 1) * self.mode

class GammaSampler:
	"""
	pareto sampler
	"""
	def __init__(self, shape, scale):
		"""
		initializer
		
		Parameters
			shape : shape
			scale : scale
		"""
		self.shape = shape
		self.scale = scale

	def isNumeric(self):
		return True

	def sample(self):
		"""
		samples value
		"""
		return np.random.gamma(self.shape, self.scale)

class GaussianRejectSampler:
	"""
	gaussian sampling based on rejection sampling
	"""
	def __init__(self, mean, stdDev):
		"""
		initializer
		
		Parameters
			mean : mean
			stdDev : std deviation
		"""
		self.mean = mean
		self.stdDev = stdDev
		self.xmin = mean - 3 * stdDev
		self.xmax = mean + 3 * stdDev
		self.ymin = 0.0
		self.fmax = 1.0 / (math.sqrt(2.0 * 3.14) * stdDev)
		self.ymax = 1.05 * self.fmax
		self.sampleAsInt = False
		
	def isNumeric(self):
		return True
	
	def sampleAsIntValue(self):
		"""
		sample as int value
		"""
		self.sampleAsInt = True

	def sample(self):
		"""
		samples value
		"""
		done = False
		samp = 0
		while not done:
			x = randomFloat(self.xmin, self.xmax)
			y = randomFloat(self.ymin, self.ymax)
			f = self.fmax * math.exp(-(x - self.mean) * (x - self.mean) / (2.0 * self.stdDev * self.stdDev))
			if (y < f):
				done = True
				samp = x
		if self.sampleAsInt:
			samp = int(samp)
		return samp

class DiscreteRejectSampler:
	"""
	non parametric sampling for discrete values  using given distribution based 
	on rejection sampling	
	"""
	def __init__(self,  xmin, xmax, step, *values):
		"""
		initializer
		
		Parameters
			xmin : min  value
			xmax : max  value
			step : discrete step
			values : distr values
		"""
		self.xmin = xmin
		self.xmax = xmax
		self.step = step
		self.distr = values
		if (len(self.distr) == 1):
			self.distr = self.distr[0]	
		numSteps = int((self.xmax - self.xmin) / self.step)
		#print("{:.3f} {:.3f} {:.3f} {}".format(self.xmin, self.xmax, self.step, numSteps))
		assert len(self.distr)	== numSteps + 1, "invalid number of distr values expected {}".format(numSteps + 1)
		self.ximin = 0
		self.ximax = numSteps
		self.pmax = float(max(self.distr))

	def isNumeric(self):
		return True

	def sample(self):
		"""
		samples value
		"""
		done = False
		samp = None
		while not done:
			xi = randint(self.ximin, self.ximax)
			#print(formatAny(xi, "xi"))
			ps = randomFloat(0.0, self.pmax)
			pa = self.distr[xi]
			if ps < pa:
				samp = self.xmin + xi  * self.step
				done = True
		return samp


class TriangularRejectSampler:
	"""
	non parametric sampling using triangular distribution based on rejection sampling	
	"""
	def __init__(self, xmin, xmax, vertexValue, vertexPos=None):
		"""
		initializer
		
		Parameters
			xmin : min  value
			xmax : max  value
			vertexValue : distr value at vertex
			vertexPos : vertex pposition
		"""
		self.xmin = xmin
		self.xmax = xmax
		self.vertexValue = vertexValue
		if vertexPos: 
			assert vertexPos > xmin and vertexPos < xmax, "vertex position outside bound"
			self.vertexPos = vertexPos
		else:
			self.vertexPos = 0.5 * (xmin + xmax)
		self.s1 = vertexValue / (self.vertexPos - xmin)
		self.s2 = vertexValue / (xmax - self.vertexPos)

	def isNumeric(self):
		return True
		
	def sample(self):
		"""
		samples value
		"""
		done = False
		samp = None
		while not done:
			x = randomFloat(self.xmin, self.xmax)
			y = randomFloat(0.0, self.vertexValue)
			f = (x - self.xmin) * self.s1 if x < self.vertexPos else (self.xmax - x) * self.s2
			if (y < f):
				done = True
				samp = x
			
		return samp;	

class NonParamRejectSampler:
	"""
	non parametric sampling using given distribution based on rejection sampling	
	"""
	def __init__(self, xmin, binWidth, *values):
		"""
		initializer
		
		Parameters
			xmin : min  value
			binWidth : bin width
			values : distr values
		"""
		self.values = values
		if (len(self.values) == 1):
			self.values = self.values[0]
		self.xmin = xmin
		self.xmax = xmin + binWidth * (len(self.values) - 1)
		#print(self.xmin, self.xmax, binWidth)
		self.binWidth = binWidth
		self.fmax = 0
		for v in self.values:
			if (v > self.fmax):
				self.fmax = v
		self.ymin = 0
		self.ymax = self.fmax
		self.sampleAsInt = True

	def isNumeric(self):
		return True
		
	def sampleAsFloat(self):
		self.sampleAsInt = False
	
	def sample(self):
		"""
		samples value
		"""
		done = False
		samp = 0
		while not done:
			if self.sampleAsInt:
				x = random.randint(self.xmin, self.xmax)
				y = random.randint(self.ymin, self.ymax)
			else:
				x = randomFloat(self.xmin, self.xmax)
				y = randomFloat(self.ymin, self.ymax)
			bin = int((x - self.xmin) / self.binWidth)
			f = self.values[bin]
			if (y < f):
				done = True
				samp = x
		return samp

class JointNonParamRejectSampler:
	"""
	non parametric sampling using given distribution based on rejection sampling	
	"""
	def __init__(self, xmin, xbinWidth, xnbin, ymin, ybinWidth, ynbin, *values):
		"""
		initializer
		
		Parameters
			xmin : min  value for x
			xbinWidth : bin width for x
			xnbin : no of bins for x
			ymin : min  value for y
			ybinWidth : bin width for y
			ynbin : no of bins for y
			values : distr values
		"""
		self.values = values
		if (len(self.values) == 1):
			self.values = self.values[0]
		assert len(self.values) ==  xnbin * ynbin, "wrong number of values for joint distr"
		self.xmin = xmin
		self.xmax = xmin + xbinWidth * xnbin
		self.xbinWidth = xbinWidth
		self.ymin = ymin
		self.ymax = ymin + ybinWidth * ynbin
		self.ybinWidth = ybinWidth
		self.pmax = max(self.values)
		self.values = np.array(self.values).reshape(xnbin, ynbin)

	def isNumeric(self):
		return True

	def sample(self):
		"""
		samples value
		"""
		done = False
		samp = 0
		while not done:
			x = randomFloat(self.xmin, self.xmax)
			y = randomFloat(self.ymin, self.ymax)
			xbin = int((x - self.xmin) / self.xbinWidth)
			ybin = int((y - self.ymin) / self.ybinWidth)
			ap = self.values[xbin][ybin]
			sp = randomFloat(0.0, self.pmax)
			if (sp < ap):
				done = True
				samp = [x,y]
		return samp


class JointNormalSampler:
	"""
	joint normal sampler	
	"""
	def __init__(self, *values):
		"""
		initializer
		
		Parameters
			values : 2 mean values followed by 4 values for covar matrix
		"""
		lvalues = list(values)
		assert len(lvalues) == 6, "incorrect number of arguments for joint normal sampler"
		mean = lvalues[:2]
		self.mean = np.array(mean)
		sd = lvalues[2:]
		self.sd = np.array(sd).reshape(2,2)

	def isNumeric(self):
		return True
		
	def sample(self):
		"""
		samples value
		"""
		return list(np.random.multivariate_normal(self.mean, self.sd))
		
		
class MultiVarNormalSampler:
	"""
	muti variate normal sampler	
	"""
	def __init__(self, numVar, *values):
		"""
		initializer
		
		Parameters
			numVar : no of variables
			values : numVar mean values followed by numVar x numVar values for covar matrix
		"""
		lvalues = list(values)
		assert len(lvalues) == numVar + numVar * numVar, "incorrect number of arguments for multi var normal sampler"
		mean = lvalues[:numVar]
		self.mean = np.array(mean)
		sd = lvalues[numVar:]
		self.sd = np.array(sd).reshape(numVar,numVar)

	def isNumeric(self):
		return True
		
	def sample(self):
		"""
		samples value
		"""
		return list(np.random.multivariate_normal(self.mean, self.sd))

class CategoricalRejectSampler:
	"""
	non parametric sampling for categorical attributes using given distribution based 
	on rejection sampling	
	"""
	def __init__(self,  *values):
		"""
		initializer
		
		Parameters
			values : list of tuples which contains a categorical value and the corresponsding distr value
		"""
		self.distr = values
		if (len(self.distr) == 1):
			self.distr = self.distr[0]
		maxv = 0
		for t in self.distr:
			if t[1] > maxv:
				maxv = t[1]
		self.maxv = maxv
		
	def sample(self):
		"""
		samples value
		"""
		done = False
		samp = ""
		while not done:
			t = self.distr[randint(0, len(self.distr)-1)]	
			d = randomFloat(0, self.maxv)	
			if (d <= t[1]):
				done = True
				samp = t[0]
		return samp


class DistrMixtureSampler:
	"""
	distr mixture sampler
	"""
	def __init__(self,  mixtureWtDistr, *compDistr):
		"""
		initializer
		
		Parameters
			mixtureWtDistr : sampler that returns index into sampler list
			compDistr : sampler list
		"""
		self.mixtureWtDistr = mixtureWtDistr
		self.compDistr = compDistr
		if (len(self.compDistr) == 1):
			self.compDistr = self.compDistr[0]
			
	def isNumeric(self):
		return True
	
	def sample(self):
		"""
		samples value
		"""
		comp = self.mixtureWtDistr.sample()
		
		#sample  sampled comp distr
		return self.compDistr[comp].sample()

class AncestralSampler:
	"""
	ancestral sampler using conditional distribution
	"""
	def __init__(self,  parentDistr, childDistr, numChildren):
		"""
		initializer
		
		Parameters
			parentDistr : parent distr
			childDistr : childdren distribution dictionary
			numChildren : no of children
		"""
		self.parentDistr = parentDistr
		self.childDistr = childDistr
		self.numChildren = numChildren
	
	def sample(self):
		"""
		samples value
		"""
		parent = self.parentDistr.sample()
		
		#sample all children conditioned on parent
		children = []
		for i in range(self.numChildren):
			key = (parent, i)
			child = self.childDistr[key].sample()
			children.append(child)
		return (parent, children)
		
class ClusterSampler:
	"""
	sample cluster and then sample member of sampled cluster
	"""
	def __init__(self,  clusters, *clustDistr):
		"""
		initializer
		
		Parameters
			clusters : dictionary clusters
			clustDistr : distr for clusters
		"""
		self.sampler = CategoricalRejectSampler(*clustDistr)
		self.clusters = clusters
	
	def sample(self):
		"""
		samples value
		"""
		cluster = self.sampler.sample()
		member = random.choice(self.clusters[cluster])
		return (cluster, member)
		
	
class MetropolitanSampler:
	"""
	metropolitan sampler	
	"""
	def __init__(self, propStdDev, min, binWidth, values):
		"""
		initializer
		
		Parameters
			propStdDev : proposal distr std dev
			min : min domain value for target distr
			binWidth : bin width
			values : target distr values
		"""
		self.targetDistr = Histogram.createInitialized(min, binWidth, values)
		self.propsalDistr = GaussianRejectSampler(0, propStdDev)
		self.proposalMixture = False
		
		# bootstrap sample
		(minv, maxv) = self.targetDistr.getMinMax()
		self.curSample = random.randint(minv, maxv)
		self.curDistr = self.targetDistr.value(self.curSample)
		self.transCount = 0
	
	def initialize(self):
		"""
		initialize
		"""
		(minv, maxv) = self.targetDistr.getMinMax()
		self.curSample = random.randint(minv, maxv)
		self.curDistr = self.targetDistr.value(self.curSample)
		self.transCount = 0
	
	def setProposalDistr(self, propsalDistr):
		"""
		set custom proposal distribution

		Parameters
			propsalDistr : proposal distribution
		"""
		self.propsalDistr = propsalDistr
	

	def setGlobalProposalDistr(self, globPropStdDev, proposalChoiceThreshold):
		"""
		set custom proposal distribution

		Parameters
			globPropStdDev : global proposal distr std deviation
			proposalChoiceThreshold : threshold for using global proposal distribution
		"""
		self.globalProposalDistr = GaussianRejectSampler(0, globPropStdDev)
		self.proposalChoiceThreshold = proposalChoiceThreshold
		self.proposalMixture = True

	def sample(self):
		"""
		samples value
		"""
		nextSample = self.proposalSample(1)
		self.targetSample(nextSample)
		return self.curSample;
	
	def proposalSample(self, skip):
		"""
		sample from proposal distribution

		Parameters
			skip : no of samples to skip
		"""
		for i in range(skip):
			if not self.proposalMixture:
				#one proposal distr
				nextSample = self.curSample + self.propsalDistr.sample()
				nextSample = self.targetDistr.boundedValue(nextSample)
			else:
				#mixture of proposal distr
				if random.random() < self.proposalChoiceThreshold:
					nextSample = self.curSample + self.propsalDistr.sample()
				else:
					nextSample = self.curSample + self.globalProposalDistr.sample()
				nextSample = self.targetDistr.boundedValue(nextSample)
				
		return nextSample
	
	def targetSample(self, nextSample):
		"""
		target sample

		Parameters
			nextSample : proposal distr sample
		"""
		nextDistr = self.targetDistr.value(nextSample)
			
		transition = False
		if nextDistr > self.curDistr:
			transition = True
		else:
			distrRatio = float(nextDistr) / self.curDistr
			if random.random() < distrRatio:
				transition = True
					
		if transition:
			self.curSample = nextSample
			self.curDistr = nextDistr
			self.transCount += 1
	
	
	def subSample(self, skip):
		"""
		sub sample

		Parameters
			skip : no of samples to skip
		"""
		nextSample = self.proposalSample(skip)
		self.targetSample(nextSample)
		return self.curSample;

	def setMixtureProposal(self, globPropStdDev, mixtureThreshold):
		"""
		mixture proposal

		Parameters
			globPropStdDev : global proposal distr std deviation
			mixtureThreshold : threshold for using global proposal distribution
		"""
		self.globalProposalDistr = GaussianRejectSampler(0, globPropStdDev)
		self.mixtureThreshold = mixtureThreshold
	
	def samplePropsal(self):
		"""
		sample from proposal distr

		"""
		if self.globalPropsalDistr is None:
			proposal = self.propsalDistr.sample()
		else:
			if random.random() < self.mixtureThreshold:
				proposal = self.propsalDistr.sample()
			else:
				proposal = self.globalProposalDistr.sample()

		return proposal

class PermutationSampler:
	"""
	permutation sampler by shuffling a list
	"""
	def __init__(self):
		"""
		initialize
		"""
		self.values = None
		self.numShuffles = None
	
	@staticmethod
	def createSamplerWithValues(values, *numShuffles):
		"""
		creator with values

		Parameters
			values : list data
			numShuffles : no of shuffles or range of no of shuffles
		"""
		sampler = PermutationSampler()
		sampler.values = values
		sampler.numShuffles = numShuffles
		return sampler
		
	@staticmethod
	def createSamplerWithRange(minv, maxv, *numShuffles):
		"""
		creator with ramge min and max
		
		Parameters
			minv : min of range
			maxv : max of range
			numShuffles : no of shuffles or range of no of shuffles
		"""
		sampler = PermutationSampler()
		sampler.values = list(range(minv, maxv + 1))
		sampler.numShuffles = numShuffles
		return sampler
		
	def sample(self):
		"""
		sample new permutation
		"""
		cloned = self.values.copy()
		shuffle(cloned, *self.numShuffles)
		return cloned
	
class SpikeyDataSampler:
	"""
	samples spikey data
	"""
	def __init__(self, intvMean, intvScale, distr, spikeValueMean, spikeValueStd, spikeMaxDuration, baseValue = 0):
		"""
		initializer
		
		Parameters
			intvMean : interval mean
			intvScale : interval std dev
			distr : type of distr for interval
			spikeValueMean : spike value mean
			spikeValueStd : spike value std dev
			spikeMaxDuration : max duration for spike
			baseValue : base or offset value
		"""
		if distr == "norm":
			self.intvSampler = NormalSampler(intvMean, intvScale)
		elif distr == "expo":
			rate = 1.0 / intvScale
			self.intvSampler = ExponentialSampler(rate)
		else:
			raise ValueError("invalid distribution")

		self.spikeSampler = NormalSampler(spikeValueMean, spikeValueStd)
		self.spikeMaxDuration = spikeMaxDuration
		self.baseValue = baseValue
		self.inSpike = False
		self.spikeCount = 0
		self.baseCount = 0
		self.baseLength = int(self.intvSampler.sample())
		self.spikeValues = list()
		self.spikeLength = None

	def sample(self):
		"""
		sample new value
		"""
		if self.baseCount <= self.baseLength:
			sampled = self.baseValue
			self.baseCount += 1
		else:
			if not self.inSpike:
				#starting spike
				spikeVal = self.spikeSampler.sample()
				self.spikeLength = sampleUniform(1, self.spikeMaxDuration)
				spikeMaxPos = 0 if self.spikeLength == 1 else sampleUniform(0, self.spikeLength-1)
				self.spikeValues.clear()
				for i in range(self.spikeLength):
					if i < spikeMaxPos:
						frac = (i + 1) / (spikeMaxPos + 1)
						frac = sampleFloatFromBase(frac, 0.1 * frac)
					elif i > spikeMaxPos:
						frac =  (self.spikeLength - i) / (self.spikeLength - spikeMaxPos)
						frac = sampleFloatFromBase(frac, 0.1 * frac)
					else:
						frac = 1.0
					self.spikeValues.append(frac * spikeVal)
					self.inSpike = True
					self.spikeCount = 0
	

			sampled = self.spikeValues[self.spikeCount]
			self.spikeCount += 1

			if self.spikeCount == self.spikeLength:
				#ending spike
				self.baseCount = 0
				self.baseLength = int(self.intvSampler.sample())
				self.inSpike = False

		return sampled


def createSampler(data):
	"""
	create sampler
	
	Parameters
		data : sampler description
	"""
	#print(data)
	items = data.split(":")
	size = len(items)
	dtype = items[-1]
	stype = items[-2]
	sampler = None
	if stype == "uniform":
		if dtype == "int":
			min = int(items[0])
			max = int(items[1])
			sampler = UniformNumericSampler(min, max)
		elif dtype == "float":
			min = float(items[0])
			max = float(items[1])
			sampler = UniformNumericSampler(min, max)
		elif dtype == "categorical":
			values = items[:-2]
			sampler = UniformCategoricalSampler(values)
	elif stype == "normal":
			mean = float(items[0])
			sd = float(items[1])
			sampler = NormalSampler(mean, sd)
			if dtype == "int":
				sampler.sampleAsIntValue()
	elif stype == "nonparam":
		if dtype == "int" or dtype == "float":
			min = int(items[0])
			binWidth = int(items[1])
			values = items[2:-2]
			values = list(map(lambda v: int(v), values))
			sampler = NonParamRejectSampler(min, binWidth, values)
			if dtype == "float":
				sampler.sampleAsFloat()
		elif dtype == "categorical":
			values = list()
			for i in range(0, size-2, 2):
				cval = items[i]
				dist = int(items[i+1])
				pair = (cval, dist)
				values.append(pair)
			sampler = CategoricalRejectSampler(values)
	elif stype == "discrete":
		vmin = int(items[0])
		vmax = int(items[1])
		step = int(items[2])
		values = list(map(lambda i : int(items[i]), range(3, len(items)-2)))
		sampler = DiscreteRejectSampler(vmin, vmax, step, values)
	else:
		raise ValueError("invalid sampler type " + dtype)
	return sampler
				 
				
			
				
				
