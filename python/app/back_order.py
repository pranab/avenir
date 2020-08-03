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
import numpy as np
from causalgraphicalmodels import CausalGraphicalModel
import matplotlib.pyplot as plt 
import sklearn as sk
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
	#i += 1
	#year = int(args[i])
	#i += 1
	#month = int(args[i])
	i += 1
	dwntm = args[i]
	i += 1
	poMarg = args[i]
	i += 1
	sampler = args[i]
	i += 1
	it = args[i]
	
	#5 years data
	it = it % 260
	year = int(it / 52)
	month = int((it % 52) / 4.33)
	#print("year {}  month {}".format(year, month))
	
	defQt = sampler.output[-1] if len(sampler.output) > 0 else 0
	if sampler.prSamples is not None:
		pdem = sampler.prSamples[0]
		
	costPerUnit = 30
	partsCostPerUnit = 12
	otherCostPerUnit = 18
	pricePerUnit = 50
	machHours = 140
	prPerMach = 70
	prCapWeek = machHours * prPerMach

	#seasonal and piece wise linear trend adjustment
	sadj = seasonal[month]
	fyear = year
	if fyear <= 2:
		tadj = trend[0] * fyear
	else:
		tadj = trend[0] * 2 + trend[1] * (fyear - 2)

	dem =  int(dem + tadj + sadj)
	pdem = int(pdem + tadj + sadj)

	#back order from parts shortage
	mpdem = pdem * (1 + poMarg)
	paOrd = int(mpdem)
	boPa = dem - paOrd if dem > paOrd else 0
	#print(formatAny(boPa, "boPa"))
	
	#back order from prod capacity limit using current demand
	rdem = dem + defQt
	if rdem <= prCapWeek:
		prCap = rdem 
		defQt = 0
	else:
		prCap = prCapWeek
		defQt = rdem - prCapWeek		
	boPcap = rdem - prCap if rdem > prCap else 0
	#print(formatAny(boPcap, "boPcap"))
	
	# max of the 2 
	bo = max(boPa, boPcap)
	#print("boPa {} boPcap {}".format(boPa, boPcap))

	#shipping cost 
	ro  = dem - bo
	sc = shipCost(ro)
	if bo > 0:
		sc += shipCost(bo)

	#revenue, cost and profit
	if boPa > 0  and isEventSampled(40):
		# back order parts from different supplier
		pc = (dem - boPa) * costPerUnit + boPa * (1.1 * partsCostPerUnit + otherCostPerUnit)
	else:
		# normal
		pc = dem * costPerUnit

	#revenue
	if bo > 0:
		#discount for back ordered quantities
		rev = ro * pricePerUnit + bo * 0.9 * pricePerUnit
	else:
		#normal
		rev = dem * pricePerUnit

	#profit
	prof = (rev - pc - sc) / dem

	#demand, prev week demand, downtime, part order margin, back order, per unit profit
	print("{},{},{:.3f},{:.3f},{},{:.2f}".format(pdem, dem, dwntm * 100, poMarg * 100, bo, prof))
	return defQt

def loadData(model, dataFile):
	"""
	loads and prepares  data
	"""
	# parameters
	fieldIndices = model.config.getStringConfig("train.data.fields")[0]
	fieldIndices = strToIntArray(fieldIndices, ",")
	featFieldIndices = model.config.getStringConfig("train.data.feature.fields")[0]
	featFieldIndices = strToIntArray(featFieldIndices, ",")

	#training data
	(data, featData) = loadDataFile(dataFile, ",", fieldIndices, featFieldIndices)
	return featData.astype(np.float32)


def infer(model, dataFile,  cindex , cvalues):
	"""
	causal inference
	"""
	#train or restore model
	useSavedModel = model.config.getBooleanConfig("predict.use.saved.model")[0]
	if useSavedModel:
		model.restoreCheckpt()
	else:
		FeedForwardNetwork.batchTrain(model) 

	featData  = loadData(model, dataFile)
	
	#scaled values
	fc = featData[:,cindex]
	me = np.mean(fc)
	sd = np.std(fc)
	print("me {:.3f}  sd {:.3f}".format(me, sd))
	scvalues = list(map(lambda v : (v - me) / sd, cvalues))
	
	#scale all data
	if (model.config.getStringConfig("common.preprocessing")[0] == "scale"):
		featData = sk.preprocessing.scale(featData)
	
	model.eval()
	for sv, v in zip(scvalues, cvalues):
		#print(featData[:5,:])
		featData[:,cindex] = sv
		#print(featData[:5,:])
		tfeatData = torch.from_numpy(featData[:,:])
		yPred = model(tfeatData)
		yPred = yPred.data.cpu().numpy()
		#print(yPred)
		yPred = yPred[:,0]
		#print(yPred[:5])
		av = yPred.mean()
		print("back order {}\tunit profit {:.2f}".format(v, av))


if __name__ == "__main__":
	op = sys.argv[1]
	if op == "simu":
		numIter = int(sys.argv[2])
		simulator = MonteCarloSimulator(numIter, border, "./log/mcsim.log", "info")
		simulator.registerNormalSampler(7000, 1000)
		simulator.registerNormalSampler(7000, 1000)
		#simulator.registerUniformSampler(1, 5)
		#simulator.registerUniformSampler(1, 12)
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
		regressor = FeedForwardNetwork(prFile)
		regressor.buildModel()
		FeedForwardNetwork.batchTrain(regressor)

	elif op == "pred":
		prFile = sys.argv[2]
		regressor = FeedForwardNetwork(prFile)
		regressor.buildModel()
		FeedForwardNetwork.predict(regressor)

	elif op == "infer":
		prFile = sys.argv[2]
		dataFile = sys.argv[3]
		bvalues = toIntList(sys.argv[4].split(","))
		regressor = FeedForwardNetwork(prFile)
		regressor.buildModel()
		infer(regressor, dataFile,  4 , bvalues)

	else:
		exitWithMsg("invalid command")
	
