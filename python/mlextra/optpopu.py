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
from opti import *
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

class EvolutionaryOptimizer(PopulationBasedOptimizer):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, configFile, domain):
		"""
		intialize
		"""
		defValues = {}
		defValues["opti.pool.size"] = (5, None)
		defValues["opti.pool.select.size"] = (3, None)
		defValues["opti.purge.score.weight"] = (0.7, None)
		defValues["opti.purge.age.scale"] = (1.0, None)
		
		super(EvolutionaryOptimizer, self).__init__(configFile, defValues, domain)

		
	def run(self):
		"""
		run optimizer
		"""
		poolSelSize = self.config.getIntConfig("opti.pool.select.size")[0]
		
		#initialize solution pool
		while len(self.pool) < self.poolSize:
			cand = self.createCandidate()
			if self.domain.isValid(cand.soln):
				self.pool.append(cand)
				self.logger.info("initial soln " + str(cand.soln))
			
		self.bestSoln = self.findBest(self.pool)
		
		#iterate
		numIter = self.config.getIntConfig("opti.num.iter")[0]
		for i in range(numIter):
			#best from a random sub set
			selected = selectRandomSubListFromList(self.pool, poolSelSize)
			bestInSel = self.findBest(selected)
			self.logger.info("subset best soln " + str(bestInSel.soln))
			
			#clone and mutate
			maxTry = 5
			tryCount = 0 
			mutStat = None
			while True:
				cloneCand = Candidate()
				cloneCand.clone(bestInSel)
				mutStat = self.mutate(cloneCand)
				if mutStat:
					if self.domain.isValid(cloneCand.soln):
						cloneCand.score = self.domain.evaluate(cloneCand.soln)
						self.logger.info("...next iteration: {} score {:.3f} ".format(i, cloneCand.score))
						break
					else:
						tryCount += 1
						if tryCount == maxTry:
							raise ValueError("invalid solution after multiple tries to mutate")
				else:
					break	
			
			#purge and add new
			if mutStat:
				self.purge()
				self.pool.append(cloneCand)
				if self.bestSoln is None:
					self.bestSoln = self.findBest(self.pool)
				elif cloneCand.score < self.bestSoln.score:
					self.bestSoln = cloneCand
					self.logger.info("better solution found")

