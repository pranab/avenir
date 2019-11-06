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
import numpy as np
import sklearn as sk
import matplotlib
import random
import jprops
import lime
import lime.lime_tabular
sys.path.append(os.path.abspath("../supv"))
import svm
import rf
import gbt
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *

class LimeInterpreter(object):
	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["inter.feature.names"] = (None, "missing feature names")
		defValues["inter.cat.feature.names"] = (None, None)
		defValues["inter.kernel.width"] = (None, None)
		defValues["inter.kernel"] = (None, None)
		defValues["inter.verbose"] = (False, None)
		defValues["inter.class.names"] = (False, None)
		defValues["inter.feature.selection"] = ("auto", None)
		defValues["inter.discretize.continuous"] = (True, None)
		defValues["inter.discretizer"] = ("quartile", None)
		defValues["inter.sample.around.instance"] = (True, None)
		defValues["inter.random.state"] = (True, None)
		defValues["explain.num.features"] = (10, None)
		defValues["explain.num.samples"] = (1000, None)

		self.config = Configuration(configFile, defValues)


	# build explainer model
	def buildModel(self, trainFeatures, catNames):
		featNames = self.config.getStringConfig("inter.feature.names")[0].split(",")
		catFeatIndices = self.config.getStringConfig("inter.cat.feature.names")[0].split(",")	
		catFeatIndices = toIntList(catFeatIndices)
		kernelWidth = self.config.getFloatConfig("inter.kernel.width")[0]
		classNames = self.config.getStringConfig("inter.class.names")[0].split(",")
		featSelection = self.config.getStringConfig("inter.feature.selection")[0]
		sampLocal = self.config.getBooleanConfig("inter.sample.around.instance")[0]
		verbose = self.config.getBooleanConfig("common.verbose")[0]

		self.explainer =  lime.lime_tabular.LimeTabularExplainer(trainFeatures, feature_names=featNames,\
			categorical_features=catFeatIndices, categorical_names=catNames, kernel_width=kernelWidth,\
			verbose=verbose,class_names=classNames,feature_selection=featSelection,sample_around_instance=sampLocal)

	# explain
	def explain(self, row, predFun):
		numFeatures = self.config.getIntConfig("explain.num.features")[0]
		numSamples = self.config.getIntConfig("explain.num.samples")[0]

		exp = self.explainer.explain_instance(row,predFun,num_features=numFeatures,num_samples=numSamples)
		return exp




