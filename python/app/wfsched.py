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

import os
import sys
import random 
import math
import numpy as np
import enquiries
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../reinf"))
from util import *
from mlutil import *
from sampler import *
from ucb import *

"""
Workforce scheduling using MAB
"""

class Worker:
	"""
	worker
	"""
	respDistr = [100, 40, 20]
	incomeDistrparams = [100, 20]
	ratingDistrparams = [3.0, 1.0]
	scoreWeights = [.4, .2, .4]
	respScores = {"pos" : 1.0, "neg" : .4, "nor" : .2}
	
	def __init__(self, name):
		"""
		initializer
			
		"""
		# response to call
		self.__setRespDistr()
		
		# daily income
		self.__setIncomeDistr()
		
		# rating
		self.__setRatingDistr()
		
		self.name = name
		self.scheduled = False
		self.selected = False
		
	def schedule(self, score):
		"""
		schedule worker
		"""
		self.selected = True
		self.resp = self.respSampler.sample()
		self.scheduled = (self.resp == "pos")
		print("worker {}  score {:.3f}  scheduled {}".format(self.name, score, self.scheduled))
		
	
	def process(self):
		"""
		process worker
		"""
		rscore = Worker.respScores[self.resp]
		
		if self.scheduled:
			income = self.incomeSampler.sample()
			if income < 50:
				iscore = 0.4
			elif income > 150:
				iscore = 1.0
			else:
				iscore = 0.4 + (income - 50) * .006
		else:
			iscore = 0
			
		if self.scheduled:
			rating = self.ratingSampler.sample()
			rascore = rating / 5.0
			rascore = rascore if rascore < 1.0 else 1.0
		else:
			rascore = 0
			
		self.score = Worker.scoreWeights[0] * rscore + Worker.scoreWeights[1] * iscore + Worker.scoreWeights[2] * rascore
		
		self.scheduled = False
		self.selected = False
		
		#shift distributions
		if isEventSampled(15):
			self.__setRespDistr()
		if isEventSampled(10):
			self.__setIncomeDistr()
		if isEventSampled(15):
			self.__setRatingDistr()

		print("worker {}  reward {:.3f}".format(self.name, self.score))
		
		return self.score
		
		
	def __setRespDistr(self):
		"""	
		shirt response distribution	
		"""
		distr = Worker.respDistr.copy()
		mutDistr(distr, 5, 3)
		#print("resp distr " + str(distr))
		self.respSampler = CategoricalRejectSampler(("pos", distr[0]), ("neg", distr[1]), ("nor", distr[2]))
	
	def __setIncomeDistr(self):
		"""
		shift income distribution
		"""
		imean = sampleFloatFromBase(Worker.incomeDistrparams[0], 0.1 * Worker.incomeDistrparams[0])
		isd = sampleFloatFromBase(Worker.incomeDistrparams[1], 0.2 * Worker.incomeDistrparams[1])
		self.incomeSampler = NormalSampler(imean, isd)
		
	def __setRatingDistr(self):
		"""
		shift rating distribution
		"""
		rmean = sampleFloatFromBase(Worker.ratingDistrparams[0], 0.1 * Worker.ratingDistrparams[0])
		rsd = sampleFloatFromBase(Worker.ratingDistrparams[1], 0.2 * Worker.ratingDistrparams[1])
		self.ratingSampler = NormalSampler(rmean, rsd)
		
class Manager:
	"""
	manager
	"""
	def __init__(self, nworker):
		"""
		initializer
			
		"""
		self.nworker = nworker	
		self.dailyDemandSamplers = [NormalSampler((.68 * nworker),5), NormalSampler((.63 * nworker),5), 
		NormalSampler((.59 * nworker),5), NormalSampler((.64 * nworker),5), NormalSampler((.69 * nworker),5),
		NormalSampler((.75 * nworker),7), NormalSampler((.81 * nworker),7)]	
		self.workers = dict()
		
		names = list()
		for _ in range(self.nworker):
			name = genID(10)
			self.workers[name] = Worker(name)
			names.append(name)
		#print(names)
		self.model = UpperConfBound(names, 20, False)
	
	def schedule(self, tm):
		"""
		schedule workers
		"""
		dow = dayOfWeek(tm)
		dem = int(self.dailyDemandSamplers[dow].sample())
		dlimit = int(.95 * self.nworker)
		dem = dem if dem < dlimit else dlimit
		print("day of week {}  workeer demand {}".format(dow, dem))
		print("starting scheduling")
		
		self.scheduled  = list()
		for _ in range(dem):
			wname, score = self.model.act()
			#print("worker selected ", wname)
			self.workers[wname].schedule(score)
			self.scheduled.append(wname)
			
	def process(self):
		"""
		process workers at end of day
		"""
		print("processing workers")
		for wname  in self.workers.keys():
			worker = self.workers[wname]
			if worker.selected:
				score = worker.process()
				self.model.setReward(wname, score)
			
##########################################################################################
if __name__ == "__main__":
	nworker = int(sys.argv[1])	
	manager = Manager(nworker)

	#query and document
	opts = ["schedule", "process",  "quit"]
	lastCh = None
	tm = pastTime(50, "d")
	tm = hourOfDayAlign(tm[1], 8)
	while True:
		ch = enquiries.choose("choose from below: ", opts)

		if ch == "schedule":
			if lastCh is None or lastCh != ch:
				manager.schedule(tm)
				tm += secInDay
				lastCh = ch
			else:
				print("next operation should be process")
			
		elif ch == "process":
			if lastCh is None or lastCh != ch:
				manager.process()
				lastCh = ch
			else:
				print("next operation should be schedule")

		elif ch == "quit":
			break
	
			