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
import statistics 
from causalgraphicalmodels import CausalGraphicalModel
import matplotlib.pyplot as plt 
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
from util import *
from sampler import *
from mcsim import *

"""
cannibalized product sale
"""

values = list()
def psale(args):
	i = 0
	q1 = int(args[i])
	q1 = q1 if q1 >= 0 else 0
	i += 1
	q2 = int(args[i])
	q2 = q2 if q2 >= 0 else 0
	i += 1
	pid1 = args[i]
	i += 1
	pid2 = args[i]
	i += 1
	ptime = args[i]
	i += 1
	iter = args[i]
	ctime = ptime + iter * 3600
	print("{},{},{}".format(pid1, ctime, q1))
	print("{},{},{}".format(pid2, ctime, q2))
	values.append(q1)


if __name__ == "__main__":
	numDays = int(sys.argv[1])
	numIter = 24 * numDays
	curTime, pastTime = pastTime(numDays, "d")
	pastTime = dayAlign(pastTime)
	tsStart = int(0.6 * numIter)
	trEnd = tsStart + 30
	trSl = -2.0
	cy = np.array([-20.0, -35.0, -55.0, -65.0, -70.0, -70.0, -50.0, -30.0, -5.0, 15.0, 35.0, 50.0,
	65.0, 65.0, 55.0, 50.0, 40.0, 30.0, 25.0, 35.0, 30.0, 20.0, 5.0, -15.0])
	cy1 = 0.8 * cy
	cy2 = 0.7 * cy1
	cy3 = 0.3 * cy1
	simulator = MonteCarloSimulator(numIter, psale, "./log/mcsim.log", "info")
	simulator.registerNormalSamplerWithTrendCycle(100, 10, 0, cy1)
	simulator.registerNormalSamplerWithTrendCycle(150, 20, 0.01, cy2)
	simulator.registerExtraArgs("DK75HUI45X", "GHT56FGT8K", pastTime)
	trSampler = NormalSamplerWithTrendCycle(100.0, 10.0, trSl , cy1)
	simulator.setSampler(0, tsStart, trSampler)
	newSampler = NormalSamplerWithTrendCycle(40, 12, 0, cy3)
	simulator.setSampler(0, trEnd, newSampler)

	simulator.run()

	drawLine(values, 250)
