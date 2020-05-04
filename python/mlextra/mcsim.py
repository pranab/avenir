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
		self.output = list()
		
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
	
	def run(self):
		"""
		run simulator
		"""
		for i in range(self.numIter):
			args = list()
			for s in self.samplers:
				arg = s.samples()
				args.append(arg)
			vOut = self.callback(args)	
			self.output.append(vOut)
	
	def getOutput(self):
		"""
		"""
		return self.output
		
	def getSum(self):
		"""
		sum
		"""
		pass
		
	def getAverage(self):
		"""
		average
		"""
		pass
		
	def getStdDev(self):
		"""
		std dev
		"""
		pass

	def getMax(self):
		"""
		max
		"""
		pass
		
	def getMin(self):
		"""
		min
		"""
		pass
		
	def getIntegral(self, drange):
		"""
		integral
		"""
		pass
	
	def getLowerTailStat(self, drange):
		"""
		lower tail stat
		"""
		pass

	def getUpperTailStat(self, drange):
		"""
		upper tail stat
		"""
		pass
