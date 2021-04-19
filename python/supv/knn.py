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
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import matplotlib
import random
import jprops
from sklearn.neighbors import KNeighborsClassifier 
from random import randint
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from bacl import *


# gradient boosting classification
class NearestNeighbor(BaseClassifier):
	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.preprocessing"] = (None, None)
		defValues["common.scaling.method"] = ("zscale", None)
		defValues["common.verbose"] = (False, None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.data.class.field"] = (None, "missing class field ordinal")
		defValues["train.num.neighbors"] = (5, None)
		defValues["train.neighbor.weight"] = ("uniform", None)
		defValues["train.neighbor.search.algo"] = ("auto", None)
		defValues["train.neighbor.search.leaf.size"] = (10, None)
		defValues["train.neighbor.dist.metric"] = ("minkowski", None)
		defValues["train.neighbor.dist.metric.pow"] = (2.0, None)
		defValues["train.success.criterion"] = ("error", None)
		defValues["train.model.save"] = (False, None)
		defValues["train.score.method"] = ("accuracy", None)
		defValues["predict.data.file"] = (None, None)
		defValues["predict.data.fields"] = (None, "missing data field ordinals")
		defValues["predict.data.feature.fields"] = (None, "missing data feature field ordinals")
		defValues["predict.use.saved.model"] = (False, None)
		
		super(NearestNeighbor, self).__init__(configFile, defValues, __name__)

	def buildModel(self):
		"""
		builds model object
		"""
		self.logger.info("...building knn classifer model")
		numNeighbors = self.config.getIntConfig("train.num.neighbors")[0]
		neighborWeight = self.config.getStringConfig("train.neighbor.weight")[0]
		searchAlgo = self.config.getStringConfig("train.neighbor.search.algo")[0]
		leafSize = self.config.getIntConfig("train.neighbor.search.leaf.size")[0]
		distMetric = self.config.getStringConfig("train.neighbor.dist.metric")[0]
		metricPow = self.config.getIntConfig("train.neighbor.dist.metric.pow")[0]
		
		model = KNeighborsClassifier(n_neighbors=numNeighbors, weights=neighborWeight, algorithm=searchAlgo, 
		leaf_size=30, p=metricPow, metric=distMetric)
		self.classifier = model
		return self.classifier
		
	def predictProb(self, recs=None):
		"""
		predict probability
		"""
		# create model
		self.prepModel()
		
		#input record
		if recs is None:
			featData = self.prepPredictData()
		else:
			if type(recs) is str:
				featData = self.prepStringPredictData(recs)
			else:
				featData = recs
			if (featData.ndim == 1):
				featData = featData.reshape(1, -1)
		
		#predict
		self.logger.info("...predicting class probability")
		clsData = self.classifier.predict_proba(featData) 
		return clsData
		
		
		