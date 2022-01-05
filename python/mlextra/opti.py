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

class Candidate(object):
	"""
	candidate solution
	"""
	counter = 0
	ops = ["insert", "delete", "replace"]
	solnSize = None
	fixedSz = True
	uniqueComp = False
	dataGroups = None
	logger = None
	
	@classmethod
	def initialize(cls, solnSize, dataGroups, logger):
		"""
		class intialization
		"""
		cls.solnSize = solnSize
		cls.logger = logger
		if (len(solnSize) == 1):
			cls.fixedSz = True
			cls.uniqueComp = False
		else:
			cls.fixedSz = False
			cls.uniqueComp = True
			cls.dataGroups = dataGroups	
    	
	def __init__(self):
		"""
		constructor
		"""
		self.soln = list()
		self.cost = None
		self.seq = Candidate.counter
		Candidate.counter += 1
		self.age = 0
	
	def clone(self, other):
		"""
		constructor
		"""
		self.soln = other.soln.copy()
		self.cost = other.cost
		self.seq = Candidate.counter
		Candidate.counter += 1
		self.age = other.age
		
	def setSoln(self, soln):
		"""
		set soln
		"""
		self.soln = soln.copy()	
		self.cost = None	

	def build(self, comp, size):
		"""
		add component
		"""
		status = True
		if Candidate.fixedSz:
			self.soln.append(comp)
		else:
			if comp in self.soln:
				status = False
			else:
				curSz = len(self.soln)
				self.soln.append(comp)
			
				if Candidate.dataGroups:
					foundGr = self.getGroup(comp)
					if foundGr:
						for c in foundGr:
							if not c == comp:
								self.soln.append(c)
						if len(self.soln) > size:
							self.soln = self.soln[:curSz]
							status = False
					
		return status
	
	def getGroup(self, comp):
		"""
		
		"""
		foundGr = None
		for gr in Candidate.dataGroups:
			if comp in gr:	
				foundGr = gr
				break
		return foundGr
		
	def mutate(self, pos, value):
		"""
		mutate to create new  soln
		"""
		status = True
		if Candidate.fixedSz:
			self.soln[pos] = value
		else:
			op = selectRandomFromList(Candidate.ops)
			backup = self.soln.copy()
			Candidate.logger.info("mutation: " + op)
			
			if op == "delete" or op == "replace":
				if Candidate.dataGroups:
					foundGr = self.getGroup(self.soln[pos])
					if foundGr:
						Candidate.logger.info("removing group")
						for v in foundGr:
							self.soln.remove(v)
					else:
						self.soln.pop(pos)
				else:
					self.soln.pop(pos)
					
			if op == "insert" or op == "replace":
				if Candidate.dataGroups:
					foundGr = self.getGroup(value)
					if foundGr:
						Candidate.logger.info("inserting group")
						for v in foundGr:
							status = self.safeInsert(v)
							if not status:
								break
					else:
						status = self.safeInsert(value)
				else:
					status = self.safeInsert(value)
					
			if not isInRange(len(self.soln), Candidate.solnSize[0], Candidate.solnSize[1]):
				Candidate.logger.info("mutation: size beyond range")
				status = False
			
			if not status:
				self.soln = backup
				
		return status	
		
	def safeInsert(self, value):
		"""
		checks and inserts
		"""
		status = True
		if value not in self.soln:
			self.soln.append(value)
		else:
			Candidate.logger.info("mutation: already exists")
			status = False
		return status
	
	def getSolnAsFloat(self):
		"""
		solution as float list
		"""
		return toFloatList(self.soln)

	def __str__(self):
		"""
		content
		"""
		strDesc = "cost: {:.3f}".format(self.cost) + " \tsoln: " + str(self.soln)
		return strDesc

class ProgressTracker(object):
	"""
	tracks optimizer progress
	"""
	def __init__(self):
		"""
		intialize
		"""
		self.tracker = list()
		self.logger = None
		
	def register(self, iter, cand):
		"""
		register current best soln
		"""
		cloned = Candidate()
		cloned.clone(cand)
		progress = (iter, cloned)
		self.tracker.append(progress)
		
	def iterSinceLastImprovement(self, iter):
		"""
		num iterations since last improvement
		"""
		return iter - self.tracker[-1][0]
		
	def lastProgessRate(self):
		"""
		cost reduction rate
		"""
		p1 = self.tracker[-2]
		p2 = self.tracker[-1]
		return (p1[1].cost - p2[1].cost) / (p2[0] - p1[0]) if len(self.tracker) > 1 else 0.0
	
	def findStats(self):
		"""
		return progress stats
		"""
		minCostDrop = None
		for i in range(len(self.tracker) - 1):
			p1 = self.tracker[i]
			p2 = self.tracker[i+1]
			costDrop = p1[1].cost - p2[1].cost
			iterCount = p2[0] - p1[0]
			rate = costDrop / iterCount
			if minCostDrop is None:
				minCostDrop = costDrop
				maxCostDrop = costDrop
				minIterCount = iterCount
				maxIterCount = iterCount
				minRate = rate
				maxRate = rate
			else:
				minCostDrop = min(costDrop, minCostDrop)
				maxCostDrop = max(costDrop, maxCostDrop)
				minIterCount = min(iterCount, minIterCount)
				maxiterCount = max(iterCount, maxIterCount)
				minRate = min(rate, minRate)
				maxRate = max(rate, maxRate)
		
		return(minCostDrop, maxCostDrop, minIterCount, maxIterCount, minRate, maxRate)
				
	def __str__(self):
		"""
		content
		"""
		content = ""
		for tracked in self.tracker:
			content = content + "iter: " + "{:04d}".format(tracked[0]) + "\t" + str(tracked[1]) + "\n"
		return content
			
		
class BaseOptimizer(object):
	"""
	base optimizer
	"""
	
	def __init__(self, configFile, defValues, domain):
		"""
		intialize
		"""
		defValues["common.verbose"] = (False, None)
		defValues["opti.solution.size"] = (None, "missing solution size")
		defValues["opti.solution.comp.size"] = (None, "missing solution component size")
		defValues["opti.solution.data.distr"] = (None, "missing solution data distribution")
		defValues["opti.solution.data.groups"] = (None, None)
		defValues["opti.num.iter"] = (100, None)
		defValues["opti.global.search.strategy"] = (None, None)
		defValues["opti.mutation.size"] = (1, None)
		defValues["opti.local.search.strategy"] = (None, None)
		defValues["opti.local.search.num.iter"] = (10, None)
		defValues["opti.performance.track.interval"] = (5, None)
		defValues["opti.performance.track.on"] = (False, None)
		defValues["opti.soln.create.max.try"] = (10, None)
		defValues["opti.soln.mutate.max.try"] = (10, None)
		
		self.config = Configuration(configFile, defValues)
		
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.domain = domain
		self.curSoln = None
		self.bestSoln = None
		self.locBestSoln = None
		self.trackedBestSoln = None
		self.globSearchStrategy = self.config.getStringConfig("opti.global.search.strategy")[0]
		self.locSearchStrategy = self.config.getStringConfig("opti.local.search.strategy")[0]
		self.mutationSize =  self.config.getIntConfig("opti.mutation.size")[0]
		self.perforTrackInterval = self.config.getIntConfig("opti.performance.track.interval")[0]
		self.numIter = self.config.getIntConfig("opti.num.iter")[0]
		self.numIterLocal = self.config.getIntConfig("opti.local.search.num.iter")[0]
		self.tracker = None
		
		#soln size and soln component data distribution
		self.solnSizes = self.config.getIntListConfig("opti.solution.size")[0]
		self.solnCompSize = self.config.getIntConfig("opti.solution.comp.size")[0]
		compDataDistr = self.config.getStringListConfig("opti.solution.data.distr")[0]
		self.compDataDistr = list(map(lambda d: createSampler(d), compDataDistr))
		
		#correlated data groups only for variable size single comp data distr solution
		dataGroups = self.config.getStringConfig("opti.solution.data.groups")[0]
		dataGroups = self.getDataGroups(dataGroups)
		
		self.varSize = len(self.solnSizes) == 2
		if not self.varSize:
			print(type(self.solnSizes[0]))
			print(type(self.solnCompSize))
			t = self.solnSizes[0] % self.solnCompSize
			assert t == 0, "soln size should be multiple of soln component size"

		self.createMaxTry = self.config.getIntConfig("opti.soln.create.max.try")[0]
		self.mutateMaxTry = self.config.getIntConfig("opti.soln.mutate.max.try")[0]

		logFilePath = self.config.getStringConfig("common.logging.file")[0]
		logLevName = self.config.getStringConfig("common.logging.level")[0]
		self.logger = createLogger(__name__, logFilePath, logLevName)
		Candidate.initialize(self.solnSizes, dataGroups, self.logger)
		
		self.trackingOn = self.config.getBooleanConfig("opti.performance.track.on")[0]
		if self.trackingOn:
			self.tracker = ProgressTracker()
			self.tracker.logger = self.logger
		
		self.solnCount = 0
		self.invalidSolnCount = 0
	# get config object
	def getConfig(self):
		return self.config
		
	def createCandidate(self):
		"""
		create new candidate soln
		"""
		tryCount = 0
		size = sampleUniform(self.solnSizes[0], self.solnSizes[1]) if self.varSize else self.solnSizes[0]
		while True:
			cand = Candidate()
			cmaxTry = 5
			built = True
			self.logger.debug("creating candidate solution")
			for j in range(size):
				value = self.sampleValue(j)
				ctryCount = 0
				while not cand.build(value, size):
					value = self.sampleValue(j)
					ctryCount += 1
					if ctryCount == cmaxTry:
						built = False
						self.logger.debug("failed to create candidate component " + str(j))
						break
						
				if not built:
					break
					
			if built:
				self.solnCount += 1
				isValid = self.domain.isValid(cand.soln)
				self.logger.debug("candidate validity " + str(isValid))
				if isValid:			
					cost = self.domain.evaluate(cand.soln)
					cand.cost = cost
					self.logger.debug("candidate cost {:.3f}".format(cost))
					break
				else:
					self.invalidSolnCount += 1
					
			tryCount += 1
			if tryCount == self.createMaxTry:
				raise ValueError("failed to create candidate solution after {} tries".format(self.createMaxTry))

		return cand

	def sampleValue(self, i):
		"""
		samples solution element value
		"""
		if self.varSize:
			value = self.compDataDistr[0].sample() 
		else: 
			ci = i % self.solnCompSize
			value = self.compDataDistr[ci].sample()
		return value

	def getDataGroups(self,data):
		"""
		co associated data
		"""
		groups = None
		if data:
			groups = list()
			items = data.split(",")
			for item in items: 
				group = strToIntArray(item, ":")
				groups.append(group)
		return groups	

	def mutate(self, cand):
		"""
		mutate by insert, delete(for var size solution only) or replace
		"""
		status = True
		self.logger.debug("before mutation soln " + str(cand.soln))
		pos = sampleUniform(0, len(cand.soln)  -1)
		value = self.sampleValue(pos)
		self.logger.debug("mutation pos {}  value {}".format(pos, value))
		tryCount = 0 
		while not cand.mutate(pos, value):
			pos = sampleUniform(0, len(cand.soln) - 1)
			value = self.sampleValue(pos)
			self.logger.debug("mutation pos {}  value {}".format(pos, value))
			tryCount += 1
			if tryCount == self.mutateMaxTry:
				status = False
				#raise ValueError("faled to mutate after multiple tries")
				self.logger.info("failed to  mutate after maximum number of tries")
				break
				
		if status:
			self.logger.debug("after mutation soln " + str(cand.soln))
		return status
		
	def multiMutate(self, cand):
		"""
		multiple mutate by insert, delete(for var size solution only) or replace
		"""
		for _ in range(self.mutationSize):
			status = self.mutate(cand)
			if not status:
				break

		return status

	def mutateAndValidate(self, cand, maxTry, clone=False):
		"""
		mutate and validate with max number of retries
		"""
		tryCount = 0 
		mutStat = None
		while True:
			if clone:
				cloneCand = Candidate()
				cloneCand.clone(cand)
			else:
				cloneCand = cand
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

			return (mutStat, cloneCand)

	def bestFromMultiMutate(cand, maxTry, numIter):
		"""
		mutate the same soluntion multiple times and find best
		"""
		bestSoln = cand
		bestCost = cand.cost
		foundBetter = False
		for i in range(numIter):
			mutStat, mutatedCand = self.mutateAndValidate(cand, 5, True)
			if mutatedCand.cost < bestCost:
				bestSoln = mutatedCand
				bestCost = mutatedCand.cost
				foundBetter = True

		return (bestSoln, foundBetter)

	def localSearch(self, cand):
		"""
		performs local search centered around a good solution
		"""
		self.logger.info("doing local search")
		bestSoln = None
		count = 0
		if self.locSearchStrategy == "centered":
			for _ in range(self.numIterLocal):
				cloneCand = self.createClone(cand)
				mvalid = self.mutate(cloneCand)
				if mvalid and self.domain.isValid(cloneCand.soln):
					count += 1
					cloneCand.cost = self.domain.evaluate(cloneCand.soln)
					if bestSoln is None or cloneCand.cost < bestSoln.cost:
						bestSoln = cloneCand
		else:
			raise ValueError("invalid local search strategy")

		if count == 0:
			self.logger.info("could not generate any valid solution in local search")
		else:
			self.logger.info("performed {} local search".format(count))

		return bestSoln

	def createClone(self, cand):
		"""
		creates cloned solution
		"""
		cloneCand = Candidate()
		cloneCand.clone(cand)
		return cloneCand

	def trackPerformance(self, iterCount):
		"""
		track performaance improvement at regular interval
		"""
		improvement = None
		if iterCount % self.perforTrackInterval == 0:
			if self.trackedBestSoln:
				if self.trackedBestSoln == self.bestSoln:
					improvement = 0.0
				else:
					improvement = (self.trackedBestSoln.cost - self.bestSoln.cost) / self.trackedBestSoln.cost
					self.trackedBestSoln = self.bestSoln
			else:
				self.trackedBestSoln = self.bestSoln
		return improvement
	
	def setBest(self, iter, bestSoln):
		"""
		sets best soln
		"""
		self.bestSoln = bestSoln
		if self.trackingOn:
			self.tracker.register(iter, bestSoln)
		self.logger.info("better solution found")

				
	def getBest(self):
		"""
		returns best solution
		"""
		return 	self.bestSoln	
			
	def getLocBest(self):
		"""
		returns best solution
		"""
		return self.locBestSoln	

class PopulationBasedOptimizer(BaseOptimizer):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, configFile, defValues, domain):
		super(PopulationBasedOptimizer, self).__init__(configFile, defValues, domain)
		self.pool = list()
		self.poolSize = self.config.getIntConfig("opti.pool.size")[0]
		self.purgeCostWt = self.config.getFloatConfig("opti.purge.cost.weight")[0]
		self.purgeAgeScale = self.config.getFloatConfig("opti.purge.age.scale")[0]
		self.fitnessDistr = None
	
	def populatePool(self):
		"""
		populate solution pool
		"""
		for i in range(self.poolSize):
			self.logger.debug("next pool memeber " + str(i))
			cand = self.createCandidate()
			self.pool.append(cand)
			self.logger.info("initial soln " + str(cand.soln))
		self.logger.info("completed initial pool creation")

	def findBest(self, candList):
		"""
		find best in candidate list
		"""
		bestCost = None
		bestSoln = None
		for cand in candList:
			if bestCost is None or cand.cost < bestCost:
				bestCost = cand.cost
				bestSoln = cand
		return bestSoln
		
	def findWorst(self, candList):
		"""
		find worst in candidate list
		"""
		worstCost = None
		worstSoln = None
		for cand in candList:
			if worstCost is None or cand.cost > worstCost:
				worstCost = cand.cost
				worstSoln = cand
		return worstSoln
	
	def findFitnessDistr(self):
		"""
		calculates fitness distribution
		"""
		worstCost = self.findWorst(self.pool).cost
		fitness = list(map(lambda c: worstCost - c.cost, self.pool))
		distr = list()
		for i, f in enumerate(fitness):
			pair = (str(i), f)
			distr.append(pair)
		self.fitnessDistr = CategoricalRejectSampler(distr)	

	def findMultBest(self, candList, size, preSorted=False):
		"""
		find best n from candidate list
		"""
		if not preSorted:
			self.sort()
		return candList[:size]

	def sort(self, candList):
		"""
		sort by ascending order of cost
		"""
		candList.sort(key=lambda c: c.cost, reverse=False)

	def tournamentSelect(self, tournSize):
		"""
		selects based on tournament algorithm
		"""
		selected = selectRandomSubListFromList(self.pool, tournSize)
		return self.findBest(selected)

		
	def fitnessDistrSelect(self, useCachedDistr):
		"""
		selects based on fitness probability algorithm
		"""
		if not useCachedDistr:
			self.findFitnessDistr()
		index = int(self.fitnessDistr.sample())
		return self.pool[i]
		
	def purge(self): 
		"""
		purge soln based on age and cost
		"""
		worstCost = 0.0
		worst = None
		for i, cand in enumerate(self.pool):
			cost = cand.cost
			age = (self.poolSize - i) * self.purgeAgeScale / self.poolSize
			aggrCost = self.purgeCostWt * cost + (1.0 - self.purgeCostWt) * age
			if aggrCost > worstCost:
				worstCost = aggrCost
				worst = i
		if self.pool[worst] == self.bestSoln:
			self.bestSoln = None
		self.pool.pop(worst)
	
	def multiPurge(self, size): 
		"""
		purge n soln based on cost
		"""
		self.pool = self.pool[:-size]

	def crossOver(self, parents):
		"""
		cross over
		"""
		children = None
		minSize = min(len(parents[0].soln), len(parents[1].soln))
		if minSize > 2:
			crossOverPoint = sampleUniform(1, minSize - 1) 
			if not self.varSize and self.solnCompSize < minSize:
				crossOverPoint = int(crossOverPoint / self.solnCompSize) * self.solnCompSize
				crossOverPoint = self.solnCompSize if crossOverPoint == 0 else crossOverPoint

			children = list()
		
			c = parents[0].soln[:crossOverPoint]
			c.extend(parents[1].soln[crossOverPoint:])
			chOne = Candidate()
			chOne.setSoln(c)
			children.append(chOne)
		
			c = parents[1].soln[:crossOverPoint]
			c.extend(parents[0].soln[crossOverPoint:])
			chTwo = Candidate()
			chTwo.setSoln(c)		
			children.append(chTwo)
			
		return children
		
		
	
	
		