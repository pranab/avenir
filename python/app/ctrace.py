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
from lstm import *

"""
Generates contact data for viral infection prediction. Also used for training LSTM model
and making predictions based on the LSTM model
"""
class TraceData:
	"""
	trace data 
	"""
	def __init__(self):
		self.trace = list()
		self.areaSamper = DiscreteRejectSampler(1, 3, 1, 60, 30, 10)
		self.targetOut = True
		self.count = 0
		self.level = list()

	def getInf(self):
		"""
		get infection status
		"""
		#contact sequence of length 5 for one person in last 15 days
		self.trace.sort(key=lambda v : v[0], reverse=True)
		tvl = 0
		mask = 0 if isEventSampled(30) else 1
		vulnerable = 1 if isEventSampled(40) else 0
		area = self.areaSamper.sample()
		
		for tr in self.trace:
			day = tr[0]
			exposure = tr[1]
			mask = mask ^ 1 if isEventSampled(10) else mask
			tr.append(mask)
			tr.append(vulnerable)
			tr.append(area)
			
			if isEventSampled(40):
				#no exposure event
				exposure = 0
				tr[1] = 0
			
			#initial viral load based on exposure and whether wearing mask
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
				#area influence	
				if area == 2:
					vld += 0.8
				elif area == 3:
					vld += 2.0

				#sigmoid viral load growth with time
				elDays = 15 - day
				vlDay = vld + elDays
				vl = math.exp(k * vlDay)
				vl = 6 * vl / (1 + vl)
				tvl += vl
				
		threshold = 7.3
		threshold = threshold if vulnerable == 1 else threshold + 0.2
		inf = 1 if tvl > threshold and isEventSampled(90) else 0
		if inf == 1:
			self.count += 1
		self.level.append(tvl)
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
		#elapsed days, exposure type, wearing mask, vulnerability, geographical area
		inf = td.getInf()
		merged = list(itertools.chain.from_iterable(td.trace))
		merged = toStrList(merged)
		st = ",".join(merged)
		if td.targetOut:
			st = st + "," + str(inf)
		print(st)
		td.trace.clear()
 		
	#print("{},{}".format(day, exposure))

if __name__ == "__main__":
	op = sys.argv[1]
	if op == "simu":
		numIter = int(sys.argv[2])
		td = TraceData()
		td.targetOut = sys.argv[3] == "y"
		simulator = MonteCarloSimulator(numIter, contact, "./log/mcsim.log", "info")
		simulator.registerUniformSampler(1, 15)
		simulator.registerDiscreteRejectSampler(1, 4, 1, 60, 24, 8, 2)
		simulator.registerExtraArgs(td)
		simulator.run()
		#print("infection count {}".format(td.count))
		#print(td.level)

	elif op == "train":
		prFile = sys.argv[2]
		classfi = LstmNetwork(prFile)
		classfi.buildModel()
		classfi.trainLstm()

	elif op == "pred":
		prFile = sys.argv[2]
		classfi = LstmNetwork(prFile)
		classfi.buildModel()
		classfi.predictLstm()
		
	else:
		exitWithMsg("invalid command")
