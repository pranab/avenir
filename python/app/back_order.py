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
sys.path.append(os.path.abspath("../supv"))
from util import *
from sampler import *
from mcsim import *
from tnn import *

"""
Manufacturing supply chain simulation

"""
seasonal = [-580, -340, 0, 370, 1250, 3230, 3980, 3760, 2770, 980, 120, -220]
trend = [250, 350]

def shipCost(quant):
	"""
	shipping and handling cost

	"""
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
	year = int(args[i])
	i += 1
	month = int(args[i])
	i += 1
	dwntm = args[i]
	i += 1
	poMarg = args[i]


	#seasonal and trend adjustment
	sadj = seasonal[month - 1]
	fyear = (year - 1) + month / 12
	if fyear <= 2:
		tadj = trend[0] * fyear
	else:
		tadj = trend[0] * 2 + trend[1] * (fyear - 2)

	dem =  int(dem + tadj + sadj)
	pdem = int(pdem + tadj + sadj)

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
	op = sys.argv[1]
	if op == "simu":
		numIter = int(sys.argv[2])
		simulator = MonteCarloSimulator(numIter, border, "./log/mcsim.log", "info")
		simulator.registerNormalSampler(9000, 1500)
		simulator.registerNormalSampler(9000, 1500)
		simulator.registerUniformSampler(1, 5)
		simulator.registerUniformSampler(1, 12)
		simulator.registerGammaSampler(1.0, .05)
		simulator.registerDiscreteRejectSampler(0.0, 0.20, 0.04, 25, 30, 18, 10, 5, 2)
		simulator.run()

	elif op == "grmo":
		bo = CausalGraphicalModel(
			nodes = ["dem", "prevDem", "partMarg", "prodDownTm", "partOrd", "boPartOrd", "prCap", "boPrCap", "bo", "profit"],
			edges =[("dem", "boPartOrd"), ("prevDem", "partOrd"), ("partMarg", "partOrd"), ("partOrd", "boPartOrd"), 
			("prodDownTm", "prCap"), ("prCap", "boPrCap") , ("dem", "boPrCap"),( "boPartOrd", "bo"), ("boPrCap", "bo")])
		bo.draw()
		plt.show()

	elif op == "train":
		prFile = sys.argv[2]
		regressor = ThreeLayerNetwork(prFile)
		regressor.buildModel()
		ThreeLayerNetwork.trainModel(regressor)

	else:
		exitWithMsg("invalid command")
	
