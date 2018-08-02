#!/Users/pranab/Tools/anaconda/bin/python

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
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
from random import randint
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from pasearch import *
from bacl import *

# gradient boosting classification
class SupportVectorMachine(BaseClassifier):

	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("train", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.scale.file.path"] = (None, "missing scale file path")
		defValues["common.preprocessing"] = ("scale", None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.data.class.field"] = (None, "missing class field ordinal")
		defValues["train.validation"] = ("kfold", None)
		defValues["train.num.folds"] = (5, None)
		defValues["train.algorithm"] = ("svc", None)
		defValues["train.kernel.function"] = ("rbf", None)
		defValues["train.poly.degree"] = (3, None)
		defValues["train.penalty"] = (1.0, None)
		defValues["train.gamma"] = ("auto", None)
		defValues["train.penalty.norm"] = ("l2", None)
		defValues["train.loss"] = ("squared_hinge", None)
		defValues["train.dual"] = (True, None)
		defValues["train.shrinking"] = (True, None)
		defValues["train.nu"] = (0.5, None)
		defValues["train.predict.probability"] = (False, None)
		defValues["train.print.sup.vectors"] = (False, None)
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
		print "...building model"
		algo = self.config.getStringConfig("train.algorithm")[0]
		kernelFun = self.config.getStringConfig("train.kernel.function")[0]
		penalty = self.config.getFloatConfig("train.penalty")[0]
		polyDegree = self.config.getIntConfig("train.poly.degree")[0]
		kernelCoeff =  self.config.getStringConfig("train.gamma")[0]
		kernelCoeff = typedValue(kernelCoeff)
		penaltyNorm = self.config.getStringConfig("train.penalty.norm")[0]
		trainLoss = self.config.getStringConfig("train.loss")[0]
		dualOpt = self.config.getBooleanConfig("train.dual")[0]
		shrinkHeuristic = self.config.getBooleanConfig("train.shrinking")[0]
		predictProb = self.config.getBooleanConfig("train.predict.probability")[0]
		supVecBound = self.config.getFloatConfig("train.nu")[0]

		if (algo == "svc"):
			if kernelFun == "poly":
				model = sk.svm.SVC(C=penalty,kernel=kernelFun,degree=polyDegree,gamma=kernelCoeff, shrinking=shrinkHeuristic, \
				probability=predictProb)
			elif kernelFun == "rbf" or kernel_fun == "sigmoid":
				model = sk.svm.SVC(C=penalty,kernel=kernelFun,gamma=kernelCoeff, shrinking=shrinkHeuristic, probability=predictProb)
			else:
				model = sk.svm.SVC(C=penalty, kernel=kernelFun, shrinking=shrinkHeuristic, probability=predictProb)
		elif (algo == "nusvc"):
			if kernelFun == "poly":
				model = sk.svm.NuSVC(nu=supVecBound, kernel=kernelFun,degree=polyDegree,gamma=kernelCoeff, shrinking=shrinkHeuristic, \
				probability=predictProb)
			elif kernelFun == "rbf" or kernelFun == "sigmoid":
				model = sk.svm.NuSVC(nu=supVecBound, kernel=kernelFun,gamma=kernelCoeff, shrinking=shrinkHeuristic, probability=predictProb)
			else:
				model = sk.svm.NuSVC(nu=supVecBound, kernel=kernelFun, shrinking=shrinkHeuristic, probability=predictProb)
		elif (algo == "linearsvc"):
			model = sk.svm.LinearSVC(penalty=penaltyNorm, loss=trainLoss, dual=dualOpt)
		else:
			print "invalid svm algorithm"
			sys.exit()
		self.classifier = model
		return self.classifier



