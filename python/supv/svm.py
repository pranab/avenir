#!/Users/pranab/Tools/anaconda/bin/python

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
		defValues["train.gamma"] = (0, None)
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
		kernelCoeff =  self.config.getFloatConfig("train.penalty")[0]

		if (algo == "svc"):
			if kernelFun == "poly":
				model = sk.svm.SVC(C=penalty,kernel=kernelFun,degree=polyDegree,gamma=kernelCoeff)
			elif kernelFun == "rbf" or kernel_fun == "sigmoid":
				model = sk.svm.SVC(C=penalty,kernel=kernelFun,gamma=kernelCoeff)
			else:
				model = sk.svm.SVC(C=penalty,kernel=kernelFun)
		elif (algo == "nusvc"):
			if kernelFun == "poly":
				model = sk.svm.NuSVC(kernel=kernelFun,degree=polyDegree,gamma=kernelCoeff)
			elif kernelFun == "rbf" or kernelFun == "sigmoid":
				model = sk.svm.NuSVC(kernel=kernelFun,gamma=kernelCoeff)
			else:
				model = sk.svm.NuSVC(kernel=kernelFun)
		elif (algo == "linearsvc"):
			model = sk.svm.LinearSVC()
		else:
			print "invalid svm algorithm"
			sys.exit()
		self.classifier = model
		return self.classifier



