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
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
from util import *
from sampler import *
from mcsim import *

"""
Manufacturing supply chain simulation

"""
seasonal = [.95, .97, 1.00, 1.03, 1.12, 1.32, 1.48, 1.41, 1.30, 1.12, 1.01, .98]

def shipCost(quant):
	if quant < 1000:
		sc = 1.6 * quant
	elif  quant < 2000:
		sc = 1.3 * quant
	else:
		sc = 1.1 * quant
	sc += 200
	return sc

def border(args):
	"""
	callback for cost calculation
	14 shifts , 10 machines 70 per machine shift 10000 products per week
	"""
	i = 0
	dem = int(args[i])
	i += 1
	pdem = int(args[i])
	i += 1
	month = int(args[i]) - 1
	i += 1
	dwntm = args[i]
	i += 1
	poMarg = args[i]


	#seasonal adjustment
	dem =  int(dem * seasonal[month])
	pdem =  int(pdem * seasonal[month])

	#back order
	paOrd = int(pdem * (1 + poMarg))
	boPa = dem - paOrd if dem > paOrd else 0
	prCap = int(14 * 10 * 70 * (1.0 - dwntm))
	boPcap = dem - prCap if dem > prCap else 0
	bo = max(boPa, boPcap)

	#shipping cost 
	ro  = dem - bo
	sc = shipCost(ro)
	if bo > 0:
		sc += shipCost(bo)

	#revenue, cost and profit
	pc = dem * 30
	rev = dem * 50
	prof = (rev - pc - sc) / dem

	#demand, prev week demand, downtime, part order margin, back order, per unit profit
	print("{},{},{:.3f},{:.3f},{},{:.3f}".format(pdem, dem, dwntm * 100, poMarg * 100, bo, prof))



if __name__ == "__main__":
	numIter = int(sys.argv[1])
	simulator = MonteCarloSimulator(numIter, border, "./log/mcsim.log", "info")
	simulator.registerNormalSampler(950, 200)
	simulator.registerNormalSampler(950, 200)
	simulator.registerUniformSampler(1, 12)
	simulator.registerGammaSampler(1.0, .05)
	simulator.registerDiscreteRejectSampler(0.0, 0.20, 0.04, 25, 30, 18, 10, 5, 2)
	simulator.run()

