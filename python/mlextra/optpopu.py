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
		defValues["opti.purge.cost.weight"] = (0.7, None)
		defValues["opti.purge.age.scale"] = (1.0, None)
		
		super(EvolutionaryOptimizer, self).__init__(configFile, defValues, domain)

		
	def run(self):
		"""
		run optimizer
		"""
		poolSelSize = self.config.getIntConfig("opti.pool.select.size")[0]
		
		#initialize solution pool
		self.populatePool()	
		self.bestSoln = self.findBest(self.pool)
		
		#iterate
		for i in range(self.numIter):
			#best from a random sub set
			bestInSel = self.tournamentSelect(poolSelSize)
			self.logger.info("tournament select best soln " + str(bestInSel.soln))
			
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
						cloneCand.cost = self.domain.evaluate(cloneCand.soln)
						self.logger.info("...next iteration: {} cost {:.3f} ".format(i, cloneCand.cost))
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
				elif cloneCand.cost < self.bestSoln.cost:
					self.bestSoln = cloneCand
					self.logger.info("better solution found")


class GeneticAlgorithmOptimizer(PopulationBasedOptimizer):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, configFile, domain):
		"""
		intialize
		"""
		defValues = {}
		defValues["opti.pool.size"] = (10, None)
		defValues["opti.mating.size"] = (5, None)
		defValues["opti.replacement.size"] = (5, None)
		defValues["opti.purge.cost.weight"] = (0.7, None)
		defValues["opti.purge.age.scale"] = (1.0, None)
		
		super(GeneticAlgorithmOptimizer, self).__init__(configFile, defValues, domain)
		
	def run(self):
		"""
		run optimizer
		"""
		self.populatePool()
		matingSize = self.config.getIntConfig("opti.mating.size")[0]
		replSize = self.config.getIntConfig("opti.replacement.size")[0]
		
		#iterate
		for i in range(self.numIter):
			matingList = findMultBest(self, candList, matingSize)
			genBest = matingList[0]
			if self.bestSoln is None or genBest.cost < self.bestSoln.cost:
				self.bestSoln = genBest
			
			#cross over	
			children = list()
			while len(children) < replSize:
				parenrs = selectRandomSubListFromList(matingList, 2)
				pair = self.crossOver(parents)
				valid = True
				for ch in pair:
					if self.domain.isValid(cloneCand.soln):
						ch.cost = self.domain.evaluate(ch.soln)
					else:
						valid = False
						break
				if valid:
					children.extend(pair)
			
			#mutate
			for ch in children:
				self.mutate(ch)
			
			#purge worst and add children for next generation
			self.multiPurge(replSize)
			self.poll.extend(children)
	
