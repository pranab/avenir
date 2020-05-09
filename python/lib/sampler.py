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
from random import randint
from util import *

from stats import Histogram

def randomFloat(low, high):
	"""
	sample float within range
	"""
	return random.random() * (high-low) + low

def randomNormSampled(mean, sd):
	"""
	sample float from normal
	"""
	return np.random.normal(mean, sd)
	
def randomNormSampledList(mean, sd, count):
	"""
	sample float list from normal 
	"""
	return np.random.normal(mean, sd, count)

def minLimit(val, min):
	"""
	min limit
	"""
	if (val < min):
		val = min
	return val

	
def rangeLimit(val, min, max):
	"""
	range limit
	"""
	if (val < min):
		val = min
	elif (val > max):
		val = max
	return val


def sampleUniform(min, max):
	"""
	sample int within range
	"""
	return randint(min, max)


def sampleFromBase(value, dev):
	"""
	sample int wrt base
	"""
	return randint(value - dev, value + dev)


def sampleFloatFromBase(value, dev):
	"""
	sample float wrt base
	"""
	return randomFloat(value - dev, value + dev)


def distrUniformWithRanndom(total, numItems, noiseLevel):
	"""
	uniformly distribute with some randomness
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


def isEventSampled(threshold, max=100):
	"""
	sample event
	"""
	return randint(0, max) < threshold


def sampleBinaryEvents(events, probPercent):
	"""
	sample binary events
	"""
	if (randint(0, 100) < probPercent):
		event = events[0]
	else:
		event = events[1]
	return event


def addNoiseNum(value, sampler):
	"""
	add noise to numeric value
	"""
	return value * (1 + sampler.sample())

	
def addNoiseCat(value, values, noise):	
	"""
	add noise to categorical value
	"""
	newValue = value
	threshold = int(noise * 100)
	if (isEventSampled(threshold)):
		newValue = selectRandomFromList(values)
	return newValue


def sampleWithReplace(data, sampSize):
	"""
	sample with replacement
	"""
	sampled = list()
	le = len(data)
	if sampSize is None:
		sampSize = le
	for i in range(sampSize):
		j = random.randint(0, le - 1)
		sampled.append(data[j])
	return sampled

class BernoulliTrialSampler:
	"""
	bernoulli trial sampler return True or False
	"""
	def __init__(self, pr):
		self.pr = pr
		
	def sample(self):
		return random.random() < self.pr
	
class PoissonSampler:
	"""
	poisson sampler returns number of events
	"""
	def __init__(self, rateOccur, maxSamp):
		self.rateOccur = rateOccur
		self.maxSamp = int(maxSamp)
		self.pmax = self.calculatePr(rateOccur)

	def calculatePr(self, numOccur):
		p = (self.rateOccur ** numOccur) * math.exp(-self.rateOccur) / math.factorial(numOccur)
		return p

	def sample(self):
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

class UniformNumericSampler:
	"""
	uniform sampler for numerical values
	"""
	def __init__(self, min, max):
		self.min = min
		self.max = max
	
	def sample(self):
		samp =	sampleUniform(self.min, self.max) if isinstance(self.min, int) else randomFloat(self.min, self.max)
		return samp	

class UniformCategoricalSampler:
	"""
	uniform sampler for categorical values
	"""
	def __init__(self, values):
		self.values = values
	
	def sample(self):
		return selectRandomFromList(self.values)	

class NormalSampler:
	"""
	normal sampler
	"""
	def __init__(self, mean, stdDev):
		self.mean = mean
		self.stdDev = stdDev

	def sample(self):
		return np.random.normal(self.mean, self.stdDev)
				
class LogNormalSampler:
	"""
	log normal sampler
	"""
	def __init__(self, mean, stdDev):
		self.mean = mean
		self.stdDev = stdDev

	def sample(self):
		return np.random.lognormal(self.mean, self.stdDev)

class ParetoSampler:
	"""
	log normal sampler
	"""
	def __init__(self, mode, shape):
		self.mode = mode
		self.shape = shape

	def sample(self):
		return (np.random.pareto(self.shape) + 1) * self.mode

class GaussianRejectSampler:
	"""
	gaussian sampling based on rejection sampling
	"""
	def __init__(self, mean, stdDev):
		self.mean = mean
		self.stdDev = stdDev
		self.xmin = mean - 3 * stdDev
		self.xmax = mean + 3 * stdDev
		self.ymin = 0.0
		self.fmax = 1.0 / (math.sqrt(2.0 * 3.14) * stdDev)
		self.ymax = 1.05 * self.fmax
		self.sampleAsInt = False
		
		
	def sampleAsInt(self):
		self.sampleAsInt = True

	def sample(self):
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
	non parametric sampling for categorical attributes using given distribution based 
	on rejection sampling	
	"""
	def __init__(self,  xmin, xmax, *values):
		self.xmin = xmin
		self.xmax = xmax
		self.distr = values
		if (len(self.distr) == 1):
			self.distr = self.distr[0]	
		assert len(self.distr)	== self.xmax - self.xmin + 1, "invalid number of distr values"
		self.pmax = float(max(self.distr))

	def sample(self):
		done = False
		samp = None
		while not done:
			x = randint(self.xmin, self.xmax)
			ps = randomFloat(0.0, self.pmax)
			pa = self.distr[x - self.xmin]
			if ps < pa:
				samp = x
				done = True
		return samp


class TriangularRejectSampler:
	"""
	non parametric sampling using triangular distribution based on rejection sampling	
	"""
	def __init__(self, xmin, xmax, vertexValue, vertexPos=None):
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
		
	def sample(self):
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
	def __init__(self, min, binWidth, *values):
		self.values = values
		if (len(self.values) == 1):
			self.values = self.values[0]
		self.xmin = min
		self.xmax = min + binWidth * (len(self.values) - 1)
		self.binWidth = binWidth
		self.fmax = 0
		for v in self.values:
			if (v > self.fmax):
				self.fmax = v
		self.ymin = 0.0
		self.ymax = self.fmax
		self.sampleAsInt = True
		
	def sampleAsFloat(self):
		self.sampleAsInt = False
	
	def sample(self):
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

class JointNormalSampler:
	"""
	joint normal sampler	
	"""
	def __init__(self, *values):
		lvalues = list(values)
		assert len(lvalues) == 6, "incorrect number of arguments for joint normal sampler"
		mean = lvalues[:2]
		self.mean = np.array(mean)
		sd = lvalues[2:]
		self.sd = np.array(sd).reshape(2,2)
		
	def sample(self):
		return list(np.random.multivariate_normal(self.mean, self.sd))
		
		

class CategoricalRejectSampler:
	"""
	non parametric sampling for categorical attributes using given distribution based 
	on rejection sampling	
	"""
	def __init__(self,  *values):
		self.distr = values
		if (len(self.distr) == 1):
			self.distr = self.distr[0]
		max = 0
		for t in self.distr:
			if t[1] > max:
				max = t[1]
		self.max = max
		
	def sample(self):
		done = False
		samp = ""
		while not done:
			t = self.distr[randint(0, len(self.distr)-1)]	
			d = random.randint(0, self.max)	
			if (d <= t[1]):
				done = True
				samp = t[0]
		return samp


class DistrMixtureSampler:
	"""
	distr mixture sampler
	"""
	def __init__(self,  mixtureWtDistr, *compDistr):
		self.mixtureWtDistr = mixtureWtDistr
		self.compDistr = compDistr
		if (len(self.compDistr) == 1):
			self.compDistr = self.compDistr[0]
	
	def sample(self):
		#sample comp wt distr
		comp = self.mixtureWtDistr.sample()
		
		#sample  sampled comp distr
		return self.compDistr[comp].sample()

class AncestralSampler:
	"""
	ancestral sampler using conditional distribution
	"""
	def __init__(self,  parentDistr, childDistr, numChildren):
		self.parentDistr = parentDistr
		self.childDistr = childDistr
		self.numChildren = numChildren
	
	def sample(self):
		#sample parent
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
		self.sampler = CategoricalRejectSampler(*clustDistr)
		self.clusters = clusters
	
	def sample(self):
		cluster = self.sampler.sample()
		member = random.choice(self.clusters[cluster])
		return (cluster, member)
		
	
class MetropolitanSampler:
	"""
	metropolitan sampler	
	"""
	def __init__(self, propStdDev, min, binWidth, values):
		self.targetDistr = Histogram.createInitialized(min, binWidth, values)
		self.propsalDistr = GaussianRejectSampler(0, propStdDev)
		self.proposalMixture = False
		
		# bootstrap sample
		(min, max) = self.targetDistr.getMinMax()
		self.curSample = random.randint(min, max)
		self.curDistr = self.targetDistr.value(self.curSample)
		self.transCount = 0
	
	# initialize	
	def initialize(self):
		(min, max) = self.targetDistr.getMinMax()
		self.curSample = random.randint(min, max)
		self.curDistr = self.targetDistr.value(self.curSample)
		self.transCount = 0
	
	# set custom proposal distribution
	def setProposalDistr(self, propsalDistr):
		self.propsalDistr = propsalDistr
	
	# set custom proposal distribution
	def setGlobalProposalDistr(self, globPropStdDev, proposalChoiceThreshold):
		self.globalProposalDistr = GaussianRejectSampler(0, globPropStdDev)
		self.proposalChoiceThreshold = proposalChoiceThreshold
		self.proposalMixture = True

	# sample	
	def sample(self):
		nextSample = self.proposalSample(1)
		self.targetSample(nextSample)
		return self.curSample;
	
	# sample from proposal distribution
	def proposalSample(self, skip):
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
	
	# target sample
	def targetSample(self, nextSample):
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
	
	
	# sub sample
	def subSample(self, skip):
		nextSample = self.proposalSample(skip)
		self.targetSample(nextSample)
		return self.curSample;

	# mixture proposal
	def setMixtureProposal(self, globPropStdDev, mixtureThreshold):
		self.globalProposalDistr = GaussianRejectSampler(0, globPropStdDev)
		self.mixtureThreshold = mixtureThreshold
	
	# sample from proposal distr
	def samplePropsal(self):
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
	permutation sampler
	"""
	def __init__(self):
		self.values = None
		self.numShuffles = None
	
	@staticmethod
	def createSamplerWithValues(values, *numShuffles):
		"""
		creator with values
		"""
		sampler = PermutationSampler()
		sampler.values = values
		sampler.numShuffles = numShuffles
		return sampler
		
	@staticmethod
	def createSamplerWithRange(min, max, *numShuffles):
		"""
		creator with ramge min and max
		"""
		sampler = PermutationSampler()
		sampler.values = list(range(min, max+1))
		sampler.numShuffles = numShuffles
		return sampler
		
	def sample(self):
		"""
		sample new permutation
		"""
		cloned = self.values.copy()
		shuffle(cloned, *self.numShuffles)
		return cloned
				
def createSampler(data):
	"""
	create sampler
	"""
	items = data.split(":")
	size = len(items)
	dtype = items[size  - 1]
	stype = items[size  - 2]
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
			sampler = GaussianRejectSampler(mean, sd)
			if dtype == "int":
				sampler.sampleAsInt()
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
			
	return sampler
				 
				
			
				
				
