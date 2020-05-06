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
import statistics 
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
from util import *
from sampler import *
from mcsim import *

"""
zhang 2 sample statistics
"""

def zcStat(data):
	ranks = data[0]
	#print("data size " + str(len(ranks)))
	l1 = data[1]
	l2 = data[2]
	l = l1 + l2
	#print("l1 {}  l2{}".format(l1, l2))
	
	s1 = 0.0
	for i in range(1, l1+1):
		s1 += math.log(l1 / (i - 0.5) - 1.0) * math.log(l / (ranks[i-1] - 0.5) - 1.0)
		
	s2 = 0.0
	for i in range(1, l2+1):
		s2 += math.log(l2 / (i - 0.5) - 1.0) * math.log(l / (ranks[l1 + i - 1] - 0.5) - 1.0)
	stat = (s1 + s2) / l
	return stat

class RankSampler(object):
	"""
	monte carlo simulator for intergation, various statistic for complex fumctions
	"""
	def __init__(self,mMin,mMax,sMin,sMax):
		self.mMin = mMin
		self.mMax = mMax
		self.sMin = sMin
		self.sMax = sMax
		
	def findRank(self,e, v):
		count =  1
		for ve in v:
			if ve < e:
				count += 1
		return count
	
	def createSample(self,m1,s1,m2,s2, noise=0):
		d1 = GaussianRejectSampler(m1, s1)
		v1 = list(map(lambda i: d1.sample(), range(0,50)))
	
		d2 =  GaussianRejectSampler(m2, s2)
		v2 = list(map(lambda i: d2.sample(), range(0,50)))
		for i in range(noise):
			swapBetweenLists(v1, v2)
			
		v1.sort()
		v2.sort()
	
		v = v1.copy()
		v.extend(v2)
		r1 = list(map(lambda e: self.findRank(e, v), v1))
		r2 = list(map(lambda e: self.findRank(e, v), v2))
		r = r1.copy()
		r.extend(r2)
	
		return r

	def sample(self):
		m1 = randomFloat(self.mMin, self.mMax)	
		s1 = randomFloat(self.sMin, self.sMax)	
		m2 = randomFloat(self.mMin, self.mMax)	
		s2 = randomFloat(self.sMin, self.sMax)	
		ranks = self.createSample(m1,s1,m2,s2, 2)
		return ranks
	
if __name__ == "__main__":
	#assert len(sys.argv) == 4, "wrong command line args"
	op = sys.argv[1]
	sampSize = int(sys.argv[2])
	halfSize = int(sampSize / 2)
	
	if op == "zc":
		numIter = int(sys.argv[3])
		
		#data = list(range(1,sampSize+1))
		#shuffle(data, int(0.02 * sampSize))
		sampler  = RankSampler(5.0, 10.0, 0.5, 1.0)
		data = sampler.createSample(5.0, .5, 10.0, 1.0, 2)
		stat = zcStat([data, halfSize, halfSize])
		print("diff distr stat {:.5f}".format(stat))		
		
		simulator = MonteCarloSimulator(numIter, zcStat)
		#shuffleLow = 0
		#shuffleHi = int(0.10 * sampSize)
		#simulator.registerRangePermutationSampler(1, sampSize, shuffleLow, shuffleHi)
		simulator.registerCustomSampler(sampler)
		simulator.registerExtraArgs(halfSize, halfSize)
		simulator.run()
		simulator.getMedian()
		critValues = simulator.getLowerTailStat(1.0)
		print("lower critical values")
		for cv in critValues:
			print("{:.5f}  {:.5f}".format(cv[0], cv[1]))

		critValues = simulator.getUpperTailStat(1.0)
		print("upper critical values")
		for cv in critValues:
			print("{:.5f}  {:.5f}".format(cv[0], cv[1]))
		print("actual stat {:.5f}".format(stat))
		simulator.drawHist()
	
	elif op == "st":
		data = list(range(1,sampSize+1))
		for s in range(0, 11):
			stats = list()
			for j in range(20):
				cl = data.copy()
				shuffle(cl, s)
				stat = zcStat([cl, halfSize, halfSize])
				stats.append(stat)
			mstat = statistics.mean(stats)
			print("shuffle {}  stat {:.5f}".format(s, mstat))
	
	elif op == "di":
		sampler  = RankSampler(5.0, 10.0, 0.5, 1.0)
		data = sampler.createSample(5.0, .5, 10.0, 1.0)
		stat = zcStat([data, halfSize, halfSize])
		print("diff distr stat {:.5f}".format(stat))

		data = sampler.createSample(5.0, .5, 5.0, .5)
		stat = zcStat([data, halfSize, halfSize])
		print("same distr stat {:.5f}".format(stat))
	else:
		raise ValueError("invalid op")
