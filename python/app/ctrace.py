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
import random
import numpy as np
import matplotlib.pyplot as plt 
import itertools
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
sys.path.append(os.path.abspath("../supv"))
from util import *
from sampler import *
from mcsim import *

"""

"""
class TraceData:
	"""
	trace data 
	"""
	def __init__(self):
		self.trace = list()

	def getInf(self):
		"""
		get infection status
		"""
		#contact sequence of length 5 for one person in last 15 days
		self.trace.sort(key=lambda v : v[0], reverse=True)
		tvl = 0
		mask = 0 if isEventSampled(30) else 1
		for tr in self.trace:
			day = tr[0]
			exposure = tr[1]
			tr.append(mask)
			if isEventSampled(40):
				#no exposure event
				exposure = 0
				tr[1] = 0
				
			if exposure == 0:
				pass
			elif exposure == 1:
				#none of the 3 factors
				vld = -10.0 if mask == 0 else -12.0
				k = 0.2
			elif exposure == 2:
				#1 factor
				vld = -8.0 if mask == 0 else -10.0
				k = 0.4
			elif exposure == 3:
				#2 factors
				vld = -5.0 if mask == 0 else -7.5
				k = 0.6
			elif exposure == 4:
				#all 3 factors
				vld = -3.0 if mask == 0 else -5.5
				k = 0.9
			else:
				raise ValueError("invalid exposure level")
			
			if exposure > 0:
				#sigmoid viral load growth with time
				elDays = 15 - day
				vlDay = vld + elDays
				vl = math.exp(k * vlDay)
				vl = 6 * vl / (1 + vl)
				tvl += vl
		
		inf = 1 if tvl > 5.8 and isEventSampled(90) else 0
		return inf
 		
		
def  contact(args):
	"""
	call back
	"""
	i = 0
	day = int(args[i])
	i += 1
	exposure = int(args[i])
	i += 1
	td = args[i]
	i += 2
	it = int(args[i])
	
	contact = [day, exposure]
	td.trace.append(contact)
 	
	if (it + 1) % 5 == 0:
		inf = td.getInf()
		merged = list(itertools.chain.from_iterable(td.trace))
		merged = toStrList(merged)
		st = ",".join(merged) + "," + str(inf)
		print(st)
		td.trace.clear()
 		
	#print("{},{}".format(day, exposure))

if __name__ == "__main__":
	op = sys.argv[1]
	if op == "simu":
		numIter = int(sys.argv[2])
		simulator = MonteCarloSimulator(numIter, contact, "./log/mcsim.log", "info")
		simulator.registerUniformSampler(1, 15)
		simulator.registerDiscreteRejectSampler(1, 4, 1, 60, 24, 8, 2)
		simulator.registerExtraArgs(TraceData())
		simulator.run()
