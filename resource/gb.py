#!/Users/pranab/Tools/anaconda/bin/python

# Package imports
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import matplotlib
import random
import jprops
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
from random import randint
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *


class GradinetBoostedTrees:
	def __init__(self, x):
		self.x = x

###########################################################################################
#configuration default values and exception message if default is not be used
defValues = {}
defValues["common.mode"] = ("training", None)
defValues["train.data.file"] = (None, "missing data file name")
defValues["train.data.feature.fields"] = (None, "missing data feature field ordinals")
defValues["train.data.class.field"] = (None, "missing class field ordinal")
defValues["train.validation"] = ("kfold", None)
defValues["train.num.folds"] = (5, None)
defValues["train.min.samples.split"] = ("4", None)
defValues["train.min.samples.leaf"] = ("2", None)
defValues["train.max.depth"] = (3, None)
defValues["train.max.leaf.nodes"] = (None, None)
defValues["train.max.features"] = (None, None)
defValues["train.learning.rate"] = (0.1, None)
defValues["train.num.estimators"] = (100, None)
defValues["train.subsample"] = (1.0, None)
defValues["train.loss"] = ("deviance", None)
defValues["train.random.state"] = (None, None)
defValues["train.verbose"] = (0, None)
defValues["train.warm.start"] = (False, None)
defValues["train.presort"] = ("auto", None)
defValues["train.criterion"] = ("friedman_mse", None)

# load configuration
config = Configuration(sys.argv[1], defValues)

mode = config.getStringConfig("common.mode")[0]
if mode == "train":
	dataFile = config.getStringConfig("train.data.file")[0]
	featFieldIndices = config.getStringConfig("train.data.feature.fields")[0].split(",")
	featFieldIndices = [int(a) for a in featFieldIndices]
	classFieldIndex = config.getIntConfig("train.data.class.field")[0]
	validation = config.getStringConfig("train.validation")[0]
	numFolds = config.getIntConfig("train.num.folds")[0]
	minSamplesSplit = config.getStringConfig("train.min.samples.split")[0]
	minSamplesSplit = typedValue(minSamplesSplit)
	minSamplesLeaf = config.getStringConfig("train.min.samples.leaf")[0]
	minSamplesLeaf = typedValue(minSamplesLeaf)
	#minWeightFractionLeaf = config.getFloatConfig("train.min.weight.fraction.leaf")[0]
	(maxDepth, maxLeafNodes) = config.eitherOrIntConfig("train.max.depth", "train.max.leaf.nodes")
	maxFeatures = config.getStringConfig("train.max.features")[0]
	maxFeatures = typedValue(maxFeatures)
	learningRate = config.getFloatConfig("train.learning.rate")[0]
	numEstimators = config.getIntConfig("train.num.estimators")[0]
	subsampleFraction = config.getFloatConfig("train.subsample")[0]	
	lossFun = config.getStringConfig("train.loss")[0]
	randomState = config.getIntConfig("train.random.state")[0]
	verboseOutput = config.getIntConfig("train.verbose")[0]
	warmStart = config.getBooleanConfig("train.warm.start")[0]
	presort = config.getStringConfig("train.presort")
	if (presort[1]):
		presortChoice = presort[0]
	else:
		presortChoice = presort[0].lower() == "true"
	splitCriterion = config.getStringConfig("train.criterion")[0]	
	
	gbc = GradientBoostingClassifier(loss=lossFun, learning_rate=learningRate, n_estimators=numEstimators, 
	subsample=subsampleFraction, min_samples_split=minSamplesSplit, 
	min_samples_leaf=minSamplesLeaf, min_weight_fraction_leaf=0.0, max_depth=maxDepth,  
	init=None, random_state=randomState, max_features=maxFeatures, verbose=verboseOutput, 
	max_leaf_nodes=maxLeafNodes, warm_start=warmStart, presort=presortChoice)
	
	
	
	
	