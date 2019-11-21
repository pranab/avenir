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
import sklearn.linear_model
import matplotlib
import random
import jprops
from sklearn.linear_model import LogisticRegression
from random import randint
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from pasearch import *
from bacl import *

# logistic regression classification
class LogisticRegressionDiscriminant(BaseClassifier):

	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("train", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.scale.file.path"] = (None, "missing scale file path")
		defValues["common.preprocessing"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.data.class.field"] = (None, "missing class field ordinal")
		defValues["train.validation"] = ("kfold", None)
		defValues["train.num.folds"] = (5, None)
		defValues["train.penalty"] = ("l2", None)
		defValues["train.dual"] = (False, None)
		defValues["train.tolerance"] = (0.0001, None)
		defValues["train.regularization"] = (1.0, None)
		defValues["train.fit.intercept"] = (True, None)
		defValues["train.intercept.scaling"] = (1.0, None)
		defValues["train.class.weight"] = (None, None)
		defValues["train.random.state"] = (None, None)
		defValues["train.solver"] = ("liblinear", None)
		defValues["train.max.iter"] = (100, None)
		defValues["train.multi.class"] = ("ovr", None)
		defValues["train.verbose"] = (0, None)
		defValues["train.warm.start"] = (False, None)
		defValues["train.num.jobs"] = (None, None)
		defValues["train.l1.ratio"] = (None, None)
		defValues["train.success.criterion"] = ("error", None)
		defValues["train.model.save"] = (False, None)
		defValues["train.score.method"] = ("accuracy", None)
		defValues["train.search.param.strategy"] = (None, None)
		defValues["train.search.params"] = (None, None)
		defValues["predict.data.file"] = (None, None)
		defValues["predict.data.fields"] = (None, "missing data field ordinals")
		defValues["predict.data.feature.fields"] = (None, "missing data feature field ordinals")
		defValues["predict.use.saved.model"] = (False, None)
		defValues["validate.data.file"] = (None, "missing validation data file")
		defValues["validate.data.fields"] = (None, "missing validation data field ordinals")
		defValues["validate.data.feature.fields"] = (None, "missing validation data feature field ordinals")
		defValues["validate.data.class.field"] = (None, "missing class field ordinal")
		defValues["validate.use.saved.model"] = (False, None)
		defValues["validate.score.method"] = ("accuracy", None)

		super(SupportVectorMachine, self).__init__(configFile, defValues)

	# builds model object
	def buildModel(self):
		print ("...building logistic regression model")
		penalty = self.config.getStringConfig("train.penalty")[0]
		dual = self.config.getBooleanConfig("train.dual")[0]
		tol = self.config.getFloatConfig("train.tolerance")[0]
		c = self.config.getFloatConfig("train.regularization")[0]
		fitIntercept = self.config.getBooleanConfig("train.fit.intercept")[0]
		interceptScaling = self.config.getFloatConfig("train.intercept.scaling")[0]
		classWeight = self.config.getStringConfig("train.class.weight")[0]
		randomState = self.config.getIntConfig("train.random.state")[0]
		solver = self.config.getStringConfig("train.solver")[0]
		maxIter = self.config.getIntConfig("train.max.iter")[0]
		multiClass = self.config.getStringConfig("train.multi.class")[0]
		verbos = self.config.getIntConfig("train.verbose")[0]
		warmStart = self.config.getBooleanConfig("train.warm.start")[0]
		nJobs = self.config.getIntConfig("train.num.jobs")[0]
		l1Ratio = self.config.getFloatConfig("train.l1.ratio")[0]
		
		self.classifier = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=c, fit_intercept=fitIntercept,\
			intercept_scaling=interceptScaling, class_weight=classWeight, random_state=randomState, solver=solver,\
			max_iter=maxIter, multi_class=multiClass, verbose=verbos, warm_start=warmStart, n_jobs=nJobs, l1_ratio=l1Ratio)
			
		return self.classifier
		
		
		
