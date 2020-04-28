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
	
	@classmethod
	def initialize(cls, solnSize, dataGroups):
		"""
		class intialization
		"""
		cls.solnSize = solnSize
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
		self.score = None
		self.seq = Candidate.counter
		Candidate.counter += 1
	
	def clone(self, other):
		"""
		constructor
		"""
		self.soln = other.soln.copy()
		self.score = other.score
		self.seq = Candidate.counter
		Candidate.counter += 1

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
			print("mutation: " + op)
			
			if op == "delete" or op == "replace":
				if Candidate.dataGroups:
					foundGr = self.getGroup(self.soln[pos])
					if foundGr:
						print("removing group")
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
						print("inserting group")
						for v in foundGr:
							status = self.safeInsert(v)
							if not status:
								break
					else:
						status = self.safeInsert(value)
				else:
					status = self.safeInsert(value)
					
			if not isInRange(len(self.soln), Candidate.solnSize[0], Candidate.solnSize[1]):
				print("mutation: size beyond range")
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
			print("mutation: already exists")
			status = False
		return status
	
	def __str__(self):
		"""
		content
		"""
		strDesc = "soln: " + str(self.soln) + '\n'
		strDesc = strDesc + "score: {:.3f}".format(self.score)
		return strDesc

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
		defValues["opti.solution.data.distr"] = (None, "missing solution data distribution")
		defValues["opti.solution.data.groups"] = (None, None)
		defValues["opti.num.iter"] = (100, None)
		defValues["opti.global.search.strategy"] = (None, None)
		defValues["opti.mutation.size"] = (1, None)
		defValues["opti.local.search.strategy"] = ("mutateBest", None)
		defValues["opti.performance.track.interval"] = (5, None)
		self.config = Configuration(configFile, defValues)
		
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.domain = domain
		self.curSoln = None
		self.bestSoln = None
		self.trackedBestSoln = None
		self.globSearchStrategy = self.config.getStringConfig("opti.global.search.strategy")[0]
		self.locSearchStrategy = self.config.getStringConfig("opti.local.search.strategy")[0]
		self.mutationSize =  self.config.getIntConfig("opti.mutation.size")[0]
		self.perforTrackInterval = self.config.getIntConfig("opti.performance.track.interval")[0]
		
		#soln size and soln component data distribution
		self.solnSizes = self.config.getIntListConfig("opti.solution.size")[0]
		compDataDistr = self.config.getStringListConfig("opti.solution.data.distr")[0]
		self.compDataDistr = list(map(lambda d: createSampler(d), compDataDistr))
		
		#correlated data groups only for variable size single comp data distr solution
		dataGroups = self.config.getStringConfig("opti.solution.data.groups")[0]
		dataGroups = self.getDataGroups(dataGroups)
		
		Candidate.initialize(self.solnSizes, dataGroups)
		self.varSize = len(self.solnSizes) == 2

		logFilePath = self.config.getStringConfig("common.logging.file")[0]
		logLevName = self.config.getStringConfig("common.logging.level")[0]
		self.logger = createLogger(__name__, logFilePath, logLevName)
		
	def createCandidate(self):
		"""
		create new candidate soln
		"""
		cand = Candidate()
		size = sampleUniform(self.solnSizes[0], self.solnSizes[1]) if self.varSize else self.solnSizes[0]
		maxTry = 5
		for j in range(size):
			value = self.compDataDistr[0].sample() if self.varSize else self.compDataDistr[j].sample()
			tryCount = 0
			while not cand.build(value, size):
				value = self.compDataDistr[0].sample() if self.varSize else self.compDataDistr[j].sample()
				tryCount += 1
				if tryCount == maxTry:
					break
					
		score = self.domain.evaluate(cand.soln)
		cand.score = score
		return cand

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
		self.logger.info("before mutation soln " + str(cand.soln))
		pos = sampleUniform(0, len(cand.soln)  -1)
		value = self.compDataDistr[0].sample() if self.varSize else self.compDataDistr[pos].sample()
		self.logger.info("mutation pos {}  value {}".format(pos, value))
		maxTry = 5
		tryCount = 0 
		while not cand.mutate(pos, value):
			pos = sampleUniform(0, len(cand.soln) - 1)
			value = self.compDataDistr[0].sample() if self.varSize else self.compDataDistr[pos].sample()
			self.logger.info("mutation pos {}  value {}".format(pos, value))
			tryCount += 1
			if tryCount == maxTry:
				status = False
				#raise ValueError("faled to mutate after multiple tries")
				self.logger.info("giving up on mutation")
				break
				
		if status:
			self.logger.info("after mutation soln " + str(cand.soln))
		return status
		
	def findBest(self, candList):
		"""
		find best in candidate list
		"""
		bestScore = 1000000.0
		bestSoln = None
		for cand in candList:
			if cand.score < bestScore:
				bestScore = cand.score
				bestSoln = cand
		return bestSoln
		
	
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
					improvement = (self.trackedBestSoln.score - self.bestSoln.score) / self.trackedBestSoln.score
					self.trackedBestSoln = self.bestSoln
			else:
				self.trackedBestSoln = self.bestSoln
		return improvement
				
	def getBest(self):
		"""
		returns best solution
		"""
		return 	self.bestSoln	
			
class PopulationBasedOptimizer(BaseOptimizer):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, configFile, defValues, domain):
		super(PopulationBasedOptimizer, self).__init__(configFile, defValues, domain)
		self.pool = list()
		self.poolSize = self.config.getIntConfig("opti.pool.size")[0]
		self.purgeScoreWt = self.config.getFloatConfig("opti.purge.score.weight")[0]
		self.purgeAgeScale = self.config.getFloatConfig("opti.purge.age.scale")[0]
		
	def purge(self): 
		"""
		purge soln based on age and cost
		"""
		worstScore = 0.0
		worst = None
		for i, cand in enumerate(self.pool):
			score = cand.score
			age = (self.poolSize - i) * self.purgeAgeScale / self.poolSize
			aggrScore = self.purgeScoreWt * score + (1.0 - self.purgeScoreWt) * age
			if aggrScore > worstScore:
				worstScore = aggrScore
				worst = i
		if self.pool[worst] == self.bestSoln:
			self.bestSoln = None
		self.pool.pop(worst)
	

	
	
		