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

class Action(object):
	"""
	action class for  multi arm bandit 
	"""
	
	def __init__(self, name, wsize):
		"""
		initializer
		
		Parameters
			naction : no of actions
			rwindow : reward window size
		"""
		self.name = name
		self.available = True
		self.rwindow = RollingStat(wsize)
		self.nplay = 0

	def makeAvailable(self, available):
		"""
		sets available flag
		
		Parameters
			available : available flag
		"""
		self.available = available
	
	def addReward(self, reward):
		"""
		add reward
		
		Parameters
			reward : reward value
		"""
		self.rwindow.add(reward)	
		
	def getAverageReward(self):
		"""
		get average reward
		"""
		rav, rsd = self.rwindow.getStat()
		return rav
	
	def __str__(self):
		"""
		content
		"""
		desc = "name {}  available {}  window size {}  no of play {}".format(self.name, self.available, self.rwindow.getSize(), self.nplay)
		return desc
	