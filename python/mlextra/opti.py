#!/usr/bin/python

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
import sklearn as sk
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
	
	@classmethod
	def initialize(cls, solnSize):
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
    		
    	
	def __init__(self):
		"""
		constructor
		"""
		self.soln = list()
		self.score = None
		self.seq = Candidate.counter
		Candidate.counter += 1
	
	def __init__(self, other):
		"""
		constructor
		"""
		self.soln = other.soln
		self.score = other.score
		self.seq = Candidate.counter
		Candidate.counter += 1

	def build(self, comp):
		"""
		add component
		"""
		status = True
		if Candidate.uniqueComp and self.soln.index(comp):
			status = False
		else:
			self.soln.append(comp)
		return status
		
	def mutate(self, pos, value):
		"""
		mutate to create new  soln
		"""
		if Candidate.fixedSz:
			op = "replace"
		else:
			op = selectRandomFromList(Candidate.ops)
		status = True
		if op == "delete":
			if not Candidate.fixedSz and len(self.soln) > Candidate.solnSize[0]:
				self.soln.pop(pos)
			else:
				status = False
		elif op == "insert":
			if not Candidate.fixedSz and len(self.soln) < Candidate.solnSize[1]:
				self.soln.append(value)
			else:
				status = False
		else:
			self.soln[pos] = value
		return status	
		
		
class EvolutionaryOptimizer(object):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, configFile, domain):
		"""
		intialize
		"""
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["opti.solution.size"] = (None, "missing solution size")
		defValues["opti.solution.data.distr"] = (None, "missing solution data distribution")
		defValues["opti.pool.size"] = (5, none)
		defValues["opti.pool.select.size"] = (3, none)
		defValues["opti.num.iter"] = (100, none)
		defValues["opti.purge.score.weight"] = (0.7, none)
		defValues["opti.purge.age.scale"] = (1.0, none)

		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.domain = domain
		self.bestSoln = None
		
	def run(self):
		"""
		run optimizer
		"""
		self.pool = list()
		self.poolSize = self.config.getIntConfig("opti.pool.size")[0]
		poolSelSize = self.config.getIntConfig("opti.pool.select.size")[0]
		self.solnSizes = self.config.getIntListConfig("opti.solution.size", ",")[0]
		Candidate.initialize(self.solnSizes)
		uniqComp = len(solnSizes) == 2
		self.varSize = uniqComp
		compDataDistr = self.config.getStringListConfig("opti.solution.data.distr", ",")[0]
		self.compDataDistr = list(map(lambda d: createSampler(d), compDataDistr))
		self.purgeScoreWt = self.config.getFloatConfig("opti.purge.score.weight")[0]
		self.purgeAgeScale = self.config.getFloatConfig("opti.purge.age.scale")[0]
		
		#initialize solution pool
		for i in range(self.poolSize):
			cand = Candidate()
			size = sampleUniform(self.solnSizes[0], self.solnSizes[1]) if self.varSize else self.solnSizes[0]
			for j in range(size):
				value = self.compDataDistr[0].sample() if self.varSize else self.compDataDistr[j].sample()
				while not cand.build(value):
					value = self.compDataDistr[0].sample() if self.varSize else self.compDataDistr[j].sample()
			self.pool.append(cand)
			score = self.domain.evaluate(cand.soln)
			cand.score = score
		self.bestSoln = self.findBest(self.pool)
		
		#iterate
		numIter = self.config.getIntConfig("opti.num.iter")[0]
		for i in range(numIter):
			#best from a random sub set
			selected = selectRandomSubListFromList(pool, poolSelSize)
			bestInSel = self.findBest(selected)
			
			#clone and mutate
			clone = Candidate(bestInSel)
			clone.mutate()
			score = self.domain.evaluate(clone.soln)
			clone.score = score
			
			#purge and add new
			self.purge()
			pool.append(clone)
			self.bestSoln = self.findBest(self.pool)

	def findBest(self, candList):
		"""
		find best in candidate list
		"""
		bestScore = 1000000.0
		bestSoln = None
		for cand in candList:
			if cand.score < bestScore:
				bestScore = score
				bestSoln = cand
		return bestSoln
		
	def mutate(self, cand):
		"""
		mutate by insert, delete(for var size solution only) or replace
		"""
		pos = sampleUniform(0, len(cand.soln))
		value = self.compDataDistr[0].sample() if self.varSize else self.compDataDistr[pos].sample()
		while not cand.mutate(pos, value):
			pos = sampleUniform(0, len(cand.soln))
			value = self.compDataDistr[0].sample() if self.varSize else self.compDataDistr[pos].sample()
		
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
		self.pool.pop(i)
		
	def getBest(self):
		return 	self.bestSoln		
		
		
		