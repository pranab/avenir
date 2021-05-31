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
import torch
from torch.utils.data import DataLoader
import random
import jprops
from random import randint
import optuna
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *

"""
neural network hyper paramter tuning with ptuna
"""

def createTunerConfig(configFile):
	"""
	create tuner config pbject
	"""
	defValues = dict()
	defValues["train.num.layers"] = ([2,4], None)
	defValues["train.num.units"] = (None, "missing range of number of units")
	defValues["train.activation"] = ("relu", None)
	defValues["train.batch.normalize"] = (["true", "false"], None)
	defValues["train.dropout.prob"] = ([-0.1, 0.5], None)
	defValues["train.out.num.units"] = (None, "missing number of output units")
	defValues["train.out.activation"] = (None, "missing output activation")
	defValues["train.batch.size"] = ([16, 128], None)
	defValues["train.opt.learning.rate"] = ([.0001, .005], None)
	
	config = Configuration(configFile, defValues)
	return config

def showStudyResults(study):
	"""
	shows study results
	"""
	print("Number of finished trials: ", len(study.trials))
	print("Best trial:")
	trial = study.best_trial
	print("Value: ", trial.value)
	print("Params: ")
	for key, value in trial.params.items():
		print("  {}: {}".format(key, value))
	
	
def objective(trial, networkType, modelConfigFile, tunerConfigFile):
	"""
	optuna based hyperparamter tuning for neural network
	"""
	tConfig = createTunerConfig(tunerConfigFile)
	
	#tuning parameters
	nlayers = config.getIntListConfig("train.num.layers")[0]
	nunits = config.getIntListConfig("train.num.units")[0]
	act = config.getStringConfig("train.activation")[0]
	dropOutRange = config.getFloatListConfig("train.dropout.prob")[0]
	outNunits = config.getIntConfig("train.out.num.units")[0]
	outAct = config.getStringConfig("train.out.activation")[0]
	batchSizes = config.getIntListConfig("train.batch.size")[0]
	learningRates = config.getFloatListConfig("train.opt.learning.rate")[0]
	
	numLayers = trial.suggest_int("numLayers", nlayers[0], nlayers[1])
	
	#batch normalize on for all layers or none
	batchNormOptions = ["true", "false"]
	batchNorm = trial.suggest_categorical("batchNorm", batchNormOptions)
	
	layerConfig = ""
	maxUnits = nunits[1]
	sep = ":"
	for i in range(nlayers):
		if i < nlayers - 1:
			nunit = trial.suggest_int("numUnits_l{}".format(i), nunits[0], maxUnits)
			dropOut = trial.suggest_int("dropOut_l{}".format(i), dropOutRange[0], dropOutRange[1])
			lconfig = [str(nunit), act, batchNorm, "true", "{:.3f}".format(dropOut)]
			lconfig = sep.join(lconfig) + ","
			maxUnits = nunit
		else:
			lconfig = [str(outNunits), outAct, "false", "false", "{:.3f}".format(-0.1)]
			lconfig = sep.join(lconfig)
		layerConfig = layerConfig + lconfig

	batchSize = trial.suggest_int("batchSize", batchSizes[0], batchSizes[1])
	learningRate = trial.suggest_int("learningRate", learningRates[0], learningRates[1])
	
	#train model
	nnModel = FeedForwardNetwork(modelConfigFile)
	nnModel.setConfigParam("train.layer.data", layerConfig)
	nnModel.setConfigParam("train.batch.size", batchSize)
	nnModel.setConfigParam("train.opt.learning.rate", learningRate)
	nnModel.buildModel()
	score = FeedForwardNetwork.batchTrain(nnModel)
	return score

if __name__ == "__main__":
	assert len(sys.argv) == 5, "requires 4 command line args"
	
	networkType =  sys.argv[1]
	modelConfigFile = sys.argv[2]
	tunerConfigFile = sys.argv[3]
	numTrial = int(sys.argv[4])
	
	study = optuna.create_study()
	study.optimize(lambda trial: objective(trial, networkType, modelConfigFile, tunerConfigFile), n_trials=numTrial)
	
	showStudyResults(study)

