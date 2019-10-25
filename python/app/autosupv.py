#!/usr/bin/python

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
from random import randint
from hyperopt import tpe, hp, fmin, Trials
sys.path.append(os.path.abspath("../supv"))
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from rf import *
from svm import *
from gbt import *

def clfEvaluator(args):
	print "next evaluation"
	clf = classifiers[args["model"]]
	config = clf.getConfig()
	modelParam = args["param"]
	for paName, paValue in modelParam.items():
		clf.setConfigParam(paName, str(paValue))
		
	return clf.trainValidate()

if __name__ == "__main__":
	# config file for each classifier
	assert len(sys.argv) > 2, "missing classifier name and config files"
	maxEvals = int(sys.argv[1])
	classifiers = dict()
	clfNames = list()
	clfProb = list()
	for i in range(2, len(sys.argv)):
		items = sys.argv[i].split(":")
		clfName = items[0]
		clfConfigFile = items[1]
		print clfName + "  " + clfConfigFile

		#build classifiers
		if clfName == "rf":
			clf = RandomForest(clfConfigFile)
		elif clfName == "gbt":
			clf = GradientBoostedTrees(clfConfigFile)
		elif clfName == "svm":
			clf = SupportVectorMachine(clfConfigFile)
		else:
			raise valueError("unsupported classifier")
		classifiers[clfName] = clf	
		clfNames.append(clfName)

		#class specific probability
		if (len(items) == 3):
			clfProb.append(float(items[2]))

	# create space
	params = list()
	for clName in clfNames:
		clf = classifiers[clName]
		print "building search space for classifier " + clName
		# search space parameters
		config = clf.getConfig()
		searchParams = config.getStringConfig("train.search.params")[0].split(",")
		assert searchParams, "missing search parameter list"

		# process search params
		searchParamDetals = list()
		for searchParam in searchParams:
			paramItems = searchParam.split(":")
			extSearchParamName = str(paramItems[0])

			#get rid name component search
			paramNameItems = paramItems[0].split(".")
			del paramNameItems[1]
			searchParamName = ".".join(paramNameItems)
			searchParamType = paramItems[1]
			searchParam = (extSearchParamName, searchParamName, searchParamType)
			searchParamDetals.append(searchParam)

		# create param space for the classifier
		param = dict()
		param["model"] = clName
		modelParam = dict()
		for extSearchParamName, searchParamName, searchParamType in searchParamDetals:
			searchParamValues = config.getStringConfig(extSearchParamName)[0].split(",")

			#make feature fields coma separated
			if (extSearchParamName == "train.search.data.feature.fields"):
				searchParamValues = list(map(lambda v:v.replace(":", ",") , searchParamValues))	
					
			if (searchParamType == "string"):
				modelParam[searchParamName] = hp.choice(searchParamName,searchParamValues)
				print "string param ", searchParamName, searchParamValues
			elif (searchParamType == "int"):
				assert len(searchParamValues) == 2, "only 2 values needed for parameter range space"
				iSearchParamValues = list(map(lambda v: int(v), searchParamValues))	
				modelParam[searchParamName] = hp.choice(searchParamName,range(iSearchParamValues[0], iSearchParamValues[1]))
				print ("int param  %s %d %d") %(searchParamName, iSearchParamValues[0], iSearchParamValues[1])
			elif (searchParamType == "float"):
				assert len(searchParamValues) == 2, "only 2 values needed for parameter uniform space"
				fSearchParamValues = list(map(lambda v: float(v), searchParamValues))	
				modelParam[searchParamName] = hp.uniform(searchParamName, fSearchParamValues[0],fSearchParamValues[1])
				print ("float param  %s %.3f %.3f") %(searchParamName, fSearchParamValues[0], fSearchParamValues[1])
			else:
				raise ValueError("invalid paramter type")
		param["param"] = modelParam
		params.append(param)


	# optimize
	if (len(clfProb) == 0):
		print "unbiased classifier sampling"
		space = hp.choice("classifier", params)
	else:
		print "biased classifier sampling with given probabilities"
		prParams = zip(clfProb, params)
		space = hp.pchoice("classifier", prParams)
	trials = Trials()
	best = fmin(clfEvaluator,space, algo=tpe.suggest, trials=trials, max_evals=maxEvals)
	print best

		
