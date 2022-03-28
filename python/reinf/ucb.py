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
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *
from bandit import *

class UpperConfBound:
	"""
	upper conf bound multi arm bandit (ucb1)
	"""
	
	def __init__(self, actions, wsize, transientAction=True):
		"""
		initializer
		
		Parameters
			naction : no of actions
			rwindow : reward window size
			transientAction ; if decision involves some tied up resource it should be set True
		"""
		assertGreater(wsize, 9, "window size should be at least 10")
		self.actions = list(map(lambda aname : Action(aname, wsize), actions))
		self .totPlays = 0
		self.transientAction = transientAction
		self.tuned = False
	
	def useTuned(self):
		"""
		set to use tuned UCB model
		"""
		self.tuned =  True
			
	def act(self):
		"""
		next play return selected action
		"""
		sact = None
		scmax = 0
		for act in self.actions:
			#print(str(act))
			if act.nplay == 0:
				sact = act
				break
			else:		
				if self.transientAction or act.available:
					s1 = act.getRewardStat()
					if self.tuned:
						v = s1[1] * s1[1] + sqrt(2 * math.log(self .totPlays) / act.nplay)
						s2 = sqrt(math.log(self .totPlays) / act.nplay) + min(.25, v)
					else:
						s2 = sqrt(2 * math.log(self .totPlays) / act.nplay)
					sc  = s1[0] + s2
					
					#print("ucb score {:.3f}  {:.3f}".format(s1[0],s2))
					if sc > scmax:
						scmax = sc
						sact = act
			
		if not self.transientAction:
			sact.makeAvailable(False)
		sact.nplay += 1
		self .totPlays += 1
		#print("action selected " + str(sact))
		re = (sact.name, scmax)	
		return re
			
	def setReward(self, aname, reward):
		"""
		reward feedback for action
			
		Parameters
			act : action
			reward : reward value
		"""
		acts = list(filter(lambda act : act.name == aname, self.actions))
		assertEqual(len(acts), 1, "invalid action name")
		act = acts[0]
		act.addReward(reward)
		if not self.transientAction:
			act.makeAvailable(True)
		#print("action rewarded " + str(act))
		
	@staticmethod
	def save(model, filePath):
		"""
		saves object
				
		Parameters
			model : model object
			filePath : file path
		"""
		saveObject(model, filePath)
			
	@staticmethod
	def restore(filePath):
		"""
		restores object
				
		Parameters
			filePath : file path
		"""
		model = restoreObject(filePath)
		return model
	