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
from opti import *
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

class SimulatedAnnealing(BaseOptimizer):
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
		
		super(SimulatedAnnealing, self).__init__(configFile, defValues, domain)

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
				if math.exp((curCost - nextCost) / temp) < random.random():
					self.curSoln = mutatedCand
					curCost = self.curSoln.cost

			if i % tempUpdInterval == 0:
				if geometricCooling:
					temp *= coolingRate
				else:
					temp = (initialTemp - i * coolingRate)
				temp = 0 if temp <= 0 else temp




