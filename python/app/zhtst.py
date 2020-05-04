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
	l1 = data[1]
	l2 = data[2]
	l = l1 + l2
	
	s1 = 0.0
	for i in range(1, l1):
		s1 += math.log(l1 / (i - 0.5) - 1.0) * math.log(l / (ranks[i-1] - 0.5) - 1.0)
		
	s2 = 0.0
	for i in range(1, l2):
		s2 += math.log(l2 / (i - 0.5) - 1.0) * math.log(l / (ranks[l1 + i-1] - 0.5) - 1.0)
	stat = (s1 + s2) / l
	return stat

if __name__ == "__main__":
	assert len(sys.argv) == 4, "wrong command line args"
	op = sys.argv[1]
	sampSize = int(sys.argv[2])
	halfSize = sampSize / 2
	numIter = int(sys.argv[3])
	
	if op == "zc":
		data = list(range(1,sampSize))
		shuffle(data, int(sampSize/20))
		stat = zcStat([data, halfSize, halfSize])
		simulator = MonteCarloSimulator(numIter, zcStat)
		simulator.registerRangePermutationSampler(self, 1, sampSize)
		simulator.run()
		critValues = simulator.getLowerTailStat(2.0)
		
	else:
		raise ValueError("invalid op")
