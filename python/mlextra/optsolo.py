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
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import jprops
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from opti import *
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

class SimulatedAnnealingOptimizer(BaseOptimizer):
	"""
	optimize with simulated annealing
	"""
	def __init__(self, configFile, domain):
		"""
		intialize
		"""
		defValues = {}
		defValues["opti.initial.temo"] = (10.0, None)
		defValues["opti.temp.update.interval"] = (5, None)
		defValues["opti.cooling.rate"] = (0.9, None)
		defValues["opti.cooling.rate.geometric"] = (False, None)
		
		super(SimulatedAnnealingOptimizer, self).__init__(configFile, defValues, domain)

	def run(self):
		"""
		run optimizer
		"""
		self.curSoln = self.createCandidate()
		curCost = self.curSoln.cost
		self.bestSoln = cand.clone(self.curSoln)
		bestCost = self.bestSoln.cost

		initialTemp = temp = self.config.getFloatConfig("opti.initial.temo")[0]
		tempUpdInterval = self.config.getIntConfig("opti.temp.update.interval")[0]
		coolingRate = self.config.getFloatConfig("opti.cooling.rate")[0]
		geometricCooling = self.config.getBooleanConfig("opti.cooling.rate.geometricl")[0]

		#iterate
		for i in range(self.numIter):
			mutStat, mutatedCand = self.mutateAndValidate(self.curSoln, 5, True)
			nextCost = mutatedCand.cost
			if nextCost < curCost:
				#next cost better
				self.curSoln = mutatedCand
				curCost = self.curSoln.cost

				if mutatedCand.cost < bestCost:
					self.bestSoln = cand.clone(mutatedCand)
					bestCost = self.bestSoln.cost

			else:
				#next cost worse
				if math.exp((curCost - nextCost) / temp) > random.random():
					self.curSoln = mutatedCand
					curCost = self.curSoln.cost

			if i % tempUpdInterval == 0:
				if geometricCooling:
					temp *= coolingRate
				else:
					temp = (initialTemp - i * coolingRate)
				temp = 0 if temp <= 0 else temp


class BayesianOptimizer(BaseOptimizer):
	"""
	optimize with bayesian optimizer
	"""
	def __init__(self, configFile, domain):
		"""
		intialize
		"""
		defValues = {}
		defValues["opti.initial.model.training.size"] = (1000, None)
		defValues["opti.acquisition.samp.size"] = (100, None)
		defValues["opti.prob.acquisition.strategy"] = ("pi", None)
		defValues["opti.acquisition.lcb.mult"] = (2.0, None)
		
		super(BayesianOptimizer, self).__init__(configFile, defValues, domain)
		self.model = GaussianProcessRegressor()

	def run(self):
		"""
		run optimizer
		"""
		assert Candidare.fixedSz, "BayesianOptimizer works only for fixed size solution"

		for sampler in self.compDataDistr:
			assert sampler.isNumeric(), "BayesianOptimizer works only for numerical data"

		#inir=tial population and moel fit
		trSize = self.config.getIntConfig("opti.initial.model.training.size")[0]
		features, targets = self.createSamples(trSize)
		self.model.fit(features, targets)

		#iterate
		acqSampSize = self.config.getIntConfig("opti.acquisition.samp.size")[0]
		prAcqStrategy = self.config.getIntConfig("opti.prob.acquisition.strategy")[0]
		acqLcbMult = self.config.getFloatConfig("opti.prob.acquisition.strategy")[0]
		for i in range(self.numIter):
			ofeature, otarget = optAcquire(features, targets, acqSampSize, prAcqStrategy, acqLcbMult)
			features = np.vstack((features, [ofeature]))
			targets = np.vstack((targets, [otarget]))
			self.model.fit(features, targets)

		ix = np.argmax(targets)

	def optAcquire(features, targets, acqSampSize, prAcqStrategy, acqLcbMult):
		"""
		run optimizer
		"""
		mu = self.model.predict(features)
		best = min(mu)

		sfeatures, stargets = self.createSamples(acqSampSize)
		smu, sstd = self.model.predict(sfeatures, return_std=True)
		if prAcqStrategy == "pi":
			imp = best - smu
			z = imp / (sstd + 1E-9)
			scores = norm.cdf(z)
		elif prAcqStrategy == "ei":
			imp = best - smu
			z = imp / (sstd + 1E-9)
			scores = imp * norm.cdf(z) + sstd * norm.pdf(z)
		elif prAcqStrategy == "lcb":
			scores = smu - acqLcbMult * sstd
		else:
			raise ValueError("invalid acquisition strategy for next best candidate")
		ix = np.argmax(scores)
		sfeature = sfeatures[ix]
		starget = stargets[ix]



		return (sfeature, starget)

	def createSamples(self, size):
		"""
		sample features and targets
		"""
		features = list()
		targets = list()
		for i in range(size):
			cand = self.createCandidate()
			features.append(cand.getSolnAsFloat())
			targets.append(cand.cost)
		features = np.asarray(features)
		targets = np.asarray(targets).reshape(size, 1)
		return (features, targets)







