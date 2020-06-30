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
					self.setBest(i, self.findBest(self.pool))
				elif cloneCand.cost < self.bestSoln.cost:
					self.setBest(i, cloneCand)


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
		defValues["opti.replacement.size.var"] = (None, None)
		defValues["opti.purge.cost.weight"] = (0.7, None)
		defValues["opti.purge.age.scale"] = (1.0, None)
		defValues["opti.purge.first"] = (True, None)
		
		super(GeneticAlgorithmOptimizer, self).__init__(configFile, defValues, domain)
		
	def run(self):
		"""
		run optimizer
		"""
		self.logger.info("**** starting GeneticAlgorithmOptimizer ****")
		self.populatePool()

		matingSize = self.config.getIntConfig("opti.mating.size")[0]
		replSize = self.config.getIntConfig("opti.replacement.size")[0]
		replSizeVar = self.config.getFloatConfig("opti.replacement.size.var")[0]
		purgeFirst = self.config.getBooleanConfig("opti.purge.first")[0]
		
			
		#iterate
		self.sort(self.pool)
		preSorted = True
		for i in range(self.numIter):
			self.logger.info("next iteration " + str(i))
			matingList = self.findMultBest(self.pool, matingSize, preSorted)
			genBest = matingList[0]
			self.logger.info("current best soln cost {:.3f}".format(genBest.cost))
			if self.bestSoln is None or genBest.cost < self.bestSoln.cost:
				self.setBest(i, genBest)
				self.logger.info("new best soln found")
			
			if replSizeVar is None:
				newGenSize = replSize
				oldGenSize = replSize
			else:
				newPoolSize = 0
				while newPoolSize < (matingSize + 2):
					newGenSize = int(preturbScalar(replSize, replSizeVar))
					oldGenSize = int(preturbScalar(replSize, replSizeVar))
					newPoolSize = self.poolSize + newGenSize - oldGenSize
			self.logger.info("newGenSize  {}  oldGenSize {}".format(newGenSize, oldGenSize))

			#cross over	
			children = list()
			while len(children) < newGenSize:
				parents = selectRandomSubListFromList(matingList, 2)
				pair = self.crossOver(parents)
				if pair:
					for ch in pair:
						mutValid = self.mutate(ch)
						if mutValid and self.domain.isValid(ch.soln):
							ch.cost = self.domain.evaluate(ch.soln)
							children.append(ch)
			self.logger.info("created all children")

			
			#new generation
			if purgeFirst:
				#purge worst and add children for next generation
				self.multiPurge(oldGenSize)
				self.pool.extend(children)
				self.sort(self.pool)
				self.logger.info("purged and added  children")
			else:
				#add children for next generation and then purge worst
				self.pool.extend(children)
				self.sort(self.pool)
				self.multiPurge(oldGenSize)
				self.logger.info("added  children and purged")
			
		if self.locSearchStrategy is not None:
			locBestSoln = self.localSearch(self.bestSoln)
			self.locBestSoln = locBestSoln
			if self.locBestSoln is not None:
				if (self.locBestSoln.cost < self.bestSoln.cost):
					self.logger.info("locally search solution is best overall")
				else:
					self.logger.info("local search failed to find a better solution")


