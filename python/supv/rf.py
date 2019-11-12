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
from sklearn.ensemble import RandomForestClassifier 
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from random import randint
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from pasearch import *
from bacl import *


# gradient boosting classification
class RandomForest(BaseClassifier):
	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.preprocessing"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.data.class.field"] = (None, "missing class field ordinal")
		defValues["train.validation"] = ("kfold", None)
		defValues["train.num.folds"] = (5, None)
		defValues["train.num.trees"] = (100, None)
		defValues["train.split.criterion"] = ("gini", None)
		defValues["train.max.depth"] = (None, None)
		defValues["train.min.samples.split"] = (4, None)
		defValues["train.min.samples.leaf"] = (2, None)
		defValues["train.min.weight.fraction.leaf"] = (0, None)
		defValues["train.max.features"] = ("auto", None)
		defValues["train.max.leaf.nodes"] = (None, None)
		defValues["train.min.impurity.decrease"] = (0, None)
		defValues["train.min.impurity.split"] = (1.0e-07, None)
		defValues["train.bootstrap"] = (True, None)
		defValues["train.oob.score"] = (False, None)
		defValues["train.num.jobs"] = (1, None)
		defValues["train.random.state"] = (None, None)
		defValues["train.verbose"] = (0, None)
		defValues["train.warm.start"] = (False, None)
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
		
		super(RandomForest, self).__init__(configFile, defValues)

	# builds model object
	def buildModel(self):
		print ("...building random forest model")
		numTrees = self.config.getIntConfig("train.num.trees")[0]
		splitCriterion = self.config.getStringConfig("train.split.criterion")[0]
		maxDepth = self.config.getStringConfig("train.max.depth")[0]
		maxDepth = typedValue(maxDepth)
		minSamplesSplit = self.config.getStringConfig("train.min.samples.split")[0]
		minSamplesSplit = typedValue(minSamplesSplit)
		minSamplesLeaf = self.config.getStringConfig("train.min.samples.leaf")[0]
		minSamplesLeaf = typedValue(minSamplesLeaf)
		minWeightFractionLeaf = self.config.getFloatConfig("train.min.weight.fraction.leaf")[0]
		maxFeatures = self.config.getStringConfig("train.max.features")[0]
		maxFeatures = typedValue(maxFeatures)
		maxLeafNodes = self.config.getIntConfig("train.max.leaf.nodes")[0]
		minImpurityDecrease = self.config.getFloatConfig("train.min.impurity.decrease")[0]
		minImpurityDecrease = self.config.getFloatConfig("train.min.impurity.split")[0]
		bootstrap = self.config.getBooleanConfig("train.bootstrap")[0]
		oobScore = self.config.getBooleanConfig("train.oob.score")[0]
		numJobs = self.config.getIntConfig("train.num.jobs")[0]
		randomState = self.config.getIntConfig("train.random.state")[0]
		verbose = self.config.getIntConfig("train.verbose")[0]
		warmStart = self.config.getBooleanConfig("train.warm.start")[0]
		
		model = RandomForestClassifier(n_estimators=numTrees, criterion=splitCriterion, max_depth=maxDepth, \
		min_samples_split=minSamplesSplit, min_samples_leaf=minSamplesLeaf, min_weight_fraction_leaf=minWeightFractionLeaf, \
		max_features=maxFeatures, max_leaf_nodes=maxLeafNodes, min_impurity_decrease=minImpurityDecrease, \
		min_impurity_split=None, bootstrap=bootstrap, oob_score=oobScore, n_jobs=numJobs, random_state=randomState, \
		verbose=verbose, warm_start=warmStart, class_weight=None)
		self.classifier = model
		return self.classifier
		
	#predict probability with in memory data
	def predictProb(self, recs):
		# create model
		self.prepModel()
		
		#input record
		if type(recs) is str:
			featData = self.prepStringPredictData(recs)
		else:
			featData = recs
		if (featData.ndim == 1):
			featData = featData.reshape(1, -1)
		
		#predict
		print ("...predicting class probability")
		clsData = self.classifier.predict_proba(featData) 
		return clsData

