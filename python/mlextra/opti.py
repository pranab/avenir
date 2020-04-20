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
import lime
import lime.lime_tabular
sys.path.append(os.path.abspath("../supv"))
import svm
import rf
import gbt
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

class Candidate(object):
	"""
	candidate solution
	"""
	counter = 0
	def __init__(self, uniqueComp):
		self.soln = list()
		self.score = None
		self.uniqueComp = uniqueComp
		self.seq = counter
		counter += 1
	
	def build(self, comp):
		status = True
		if self.uniqueComp and self.soln.index(comp):
			status = False
		else:
			self.soln.append(comp)
		return status
		
	def mutate(self, op, value = None):
		status = True
		pos = sampleUniform(0, len(soln))
		if op == "delete":
			soln.pop(pos)
		else:
			if self.uniqueComp and self.soln.index(value):
				status = False
			else:
				if op == "insert":
					soln.append(value)
				else:
					soln[pos] = value
		return status	
		
		
class EvolutionaryOptimizer(object):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, configFile, domain):
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["opti.solution.size"] = (None, "missing solution size")
		defValues["opti.solution.data.distr"] = (None, "missing solution data distribution")
		defValues["opti.pool.size"] = (5, none)
		defValues["opti.rand.select.size"] = (3, none)
		defValues["opti.num.iter"] = (100, none)
		

		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.domain = dpmain
		self.best = None
		
	def run(self):
		pool = list()
		poolSize = self.config.getIntConfig("opti.pool.size")[0]
		solnSizes = self.config.getIntListConfig("opti.solution.size", ",")[0]
		uniqComp = len(solnSizes) == 2
		varSize = uniqComp
		compDataDistr = self.config.getStringListConfig("opti.solution.data.distr", ",")[0]
		compDataDistr = list(map(lambda d: createSampler(d), compDataDistr))
		
		#initialize solution pool
		for i in range(poolSize):
			cand = Candidate(uniqComp)
			size = sampleUniform(solnSizes[0], solnSizes[1]) if varSize else solnSizes[0]
			for j in range(size):
				value = compDataDistr[0].sample() if varSize else compDataDistr[j].sample()
				while not cand.build(value):
					value = compDataDistr[0].sample() if varSize else compDataDistr[j].sample()
			pool.append(cand)
			score = self.domain.evaluate(cand.soln)
			cand.score = score
		
		numIter = self.config.getIntConfig("opti.num.iter")[0]
		for i in range(numIter):
			pass
