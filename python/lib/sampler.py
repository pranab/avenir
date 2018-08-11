#!/Users/pranab/Tools/anaconda/bin/python

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

from stats import Histogram

# sample float within range
def randomFloat(low, high):
	return random.random() * (high-low) + low

# min limit
def minLimit(val, min):
	if (val < min):
		val = min
	return val

#range limit	
def rangeLimit(val, min, max):
	if (val < min):
		val = min
	elif (val > max):
		val = max
	return val

# sample int within range
def sampleUniform(min, max):
	return randint(min, max)

# sample int wrt base
def sampleFromBase(value, dev):
	return randint(value - dev, value + dev)

# sample float wrt base
def sampleFloatFromBase(value, dev):
	return randomFloat(value - dev, value + dev)

# uniformly distribute with some randomness
def distrUniformWithRanndom(total, numItems, noiseLevel):
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

# sample event
def isEventSampled(threshold, max=100):
	return randint(0, max) < threshold
		
# gaussian sampling based on rejection sampling	
class GaussianRejectSampler:
	def __init__(self, mean, stdDev):
		self.mean = mean
		self.stdDev = stdDev
		self.xmin = mean - 3 * stdDev
		self.xmax = mean + 3 * stdDev
		self.ymin = 0.0
		self.fmax = 1.0 / (math.sqrt(2.0 * 3.14) * stdDev)
		self.ymax = 1.05 * self.fmax
		
		
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
		return samp


# non parametric sampling using given distribution based on rejection sampling	
class NonParamRejectSampler:
	def __init__(self, min, binWidth, *values):
		self.xmin = min
		self.xmax = min + binWidth * (len(values) - 1)
		self.ymin = 0
		self.binWidth = binWidth
		self.values = values
		self.fmax = 0
		for v in values:
			if (v > self.fmax):
				self.fmax = v
		self.ymin = 0.0
		self.ymax = self.fmax
	
	def sample(self):
		done = False
		samp = 0
		while not done:
			x = random.randint(self.xmin, self.xmax)
			y = random.randint(self.ymin, self.ymax)
			bin = (x - self.xmin) / self.binWidth
			f = self.values[bin]
			if (y < f):
				done = True
				samp = x
		return samp

# non parametric sampling for categorical attributes using given distribution based 
# on rejection sampling	
class CategoricalRejectSampler:
	def __init__(self,  *values):
		self.distr = values
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

# sample cluster and then sample member of sampled cluster
class ClusterSampler:
	def __init__(self,  clusters, *clustDistr):
		self.sampler = CategoricalRejectSampler(*clustDistr)
		self.clusters = clusters
	
	def sample(self):
		cluster = self.sampler.sample()
		member = random.choice(self.clusters[cluster])
		return (cluster, member)
		
# metropolitan sampler		
class MetropolitanSampler:
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
