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

# gradient boosting classification
class GradientBoostedTrees:
	def __init__(self, configFile):
		defValues = {}
		defValues["common.mode"] = ("training", None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
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
		defValues["predict.data.file"] = (None, None)
		defValues["predict.data.fields"] = (None, "missing data field ordinals")
		defValues["predict.data.feature.fields"] = (None, "missing data feature field ordinals")
		
		self.config = Configuration(configFile, defValues)
		
	# get config object
	def getConfig(self):
		return self.config
	
	# train model	
	def train(self):
		#build model
		self.buildModel()
		
		# training data
		(featData, clsData) = self.prepTrainingData()
		
		#train
		print "...training model"
		self.gbcClassifier.fit(featData, clsData) 
		score = self.gbcClassifier.score(featData, clsData)  
		print "accuracy with training data %.3f" %(score)

	#train with k fold validation
	def trainAndValidate(self):
		#build model
		self.buildModel()

		# training data
		(featData, clsData) = self.prepTrainingData()
		
		#parameter
		validation = self.config.getStringConfig("train.validation")[0]
		numFolds = self.config.getIntConfig("train.num.folds")[0]
		
		#train with validation
		print "...training and cross validating model"
		scores = sk.cross_validation.cross_val_score(self.gbcClassifier, featData, clsData, cv=numFolds)
		avScore = np.mean(scores)
		print "average accuracy with k fold cross validation %.3f" %(avScore)
	 
	#predict
	def predict(self):
		# train
		self.train()
		
		# prepare test data
		featData = self.prepPredictData()
		
		#predict
		print "...predicting"
		clsData = self.gbcClassifier.predict(featData) 
		print clsData
	
	#loads and prepares training data
	def prepTrainingData(self):
		# parameters
		dataFile = self.config.getStringConfig("train.data.file")[0]
		fieldIndices = self.config.getStringConfig("train.data.fields")[0]
		if not fieldIndices is None:
			fieldIndices = strToIntArray(fieldIndices, ",")
		featFieldIndices = self.config.getStringConfig("train.data.feature.fields")[0]
		if not featFieldIndices is None:
			featFieldIndices = strToIntArray(featFieldIndices, ",")
		classFieldIndex = self.config.getIntConfig("train.data.class.field")[0]

		#training data
		(data, featData) = loadDataFile(dataFile, ",", fieldIndices, featFieldIndices)
		clsData = extrColumns(data, classFieldIndex)
		clsData = [int(a) for a in clsData]
		#print featData.shape
		#print clsData.shape
		
		return (featData, clsData)

	#loads and prepares training data
	def prepPredictData(self):
		# parameters
		dataFile = self.config.getStringConfig("predict.data.file")[0]
		if dataFile is None:
			raise ValueError("missing prediction data file")
		fieldIndices = self.config.getStringConfig("predict.data.fields")[0]
		if not fieldIndices is None:
			fieldIndices = strToIntArray(fieldIndices, ",")
		featFieldIndices = self.config.getStringConfig("predict.data.feature.fields")[0]
		if not featFieldIndices is None:
			featFieldIndices = strToIntArray(featFieldIndices, ",")

		#training data
		(data, featData) = loadDataFile(dataFile, ",", fieldIndices, featFieldIndices)
		
		return featData
	
	# builds model object
	def buildModel(self):
		print "...building model"
		# parameters
		minSamplesSplit = self.config.getStringConfig("train.min.samples.split")[0]
		minSamplesSplit = typedValue(minSamplesSplit)
		minSamplesLeaf = self.config.getStringConfig("train.min.samples.leaf")[0]
		minSamplesLeaf = typedValue(minSamplesLeaf)
		#minWeightFractionLeaf = self.config.getFloatConfig("train.min.weight.fraction.leaf")[0]
		(maxDepth, maxLeafNodes) = self.config.eitherOrIntConfig("train.max.depth", "train.max.leaf.nodes")
		maxFeatures = self.config.getStringConfig("train.max.features")[0]
		maxFeatures = typedValue(maxFeatures)
		learningRate = self.config.getFloatConfig("train.learning.rate")[0]
		numEstimators = self.config.getIntConfig("train.num.estimators")[0]
		subsampleFraction = self.config.getFloatConfig("train.subsample")[0]	
		lossFun = self.config.getStringConfig("train.loss")[0]
		randomState = self.config.getIntConfig("train.random.state")[0]
		verboseOutput = self.config.getIntConfig("train.verbose")[0]
		warmStart = self.config.getBooleanConfig("train.warm.start")[0]
		presort = self.config.getStringConfig("train.presort")
		if (presort[1]):
			presortChoice = presort[0]
		else:
			presortChoice = presort[0].lower() == "true"
		splitCriterion = self.config.getStringConfig("train.criterion")[0]	
	
		#classifier
		self.gbcClassifier = GradientBoostingClassifier(loss=lossFun, learning_rate=learningRate, n_estimators=numEstimators, 
		subsample=subsampleFraction, min_samples_split=minSamplesSplit, 
		min_samples_leaf=minSamplesLeaf, min_weight_fraction_leaf=0.0, max_depth=maxDepth,  
		init=None, random_state=randomState, max_features=maxFeatures, verbose=verboseOutput, 
		max_leaf_nodes=maxLeafNodes, warm_start=warmStart, presort=presortChoice)

###########################################################################################
gbt = GradientBoostedTrees(sys.argv[1])
mode = gbt.getConfig().getStringConfig("common.mode")[0]
if mode == "train":
	gbt.train()
elif mode == "validate":
	gbt.trainAndValidate()
elif mode == "predict":
	gbt.predict()

	
	
	