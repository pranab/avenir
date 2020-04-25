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
import random
import jprops
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../supv"))
sys.path.append(os.path.abspath("../mlextra"))
from svm import *
from rf import *
from gbt import *
from util import *
from opti import *
from sampler import *

class FeatureSelector(object):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, clfName, configFile):
		"""
		intialize
		"""
		if clfName == "rf":
			self.clf = RandomForest(clfConfigFile)
		elif clfName == "gbt":
			self.clf = GradientBoostedTrees(clfConfigFile)
		elif clfName == "svm":
			self.clf = SupportVectorMachine(clfConfigFile)
		else:
			raise valueError("unsupported classifier")
	
	def isValid(self, args):
		return True

	def evaluate(self, args):
		"""
		"""
		features = ",".join(toStrList(args))
		#print("next features: " + features)
		self.clf.setConfigParam("train.data.feature.fields", features)
		#return randomFloat(0.05, 0.25)
		score =  self.clf.trainValidate()
		print("next score {:.3f}".format(score))
		return score
		

if __name__ == "__main__":
	assert len(sys.argv) == 5, "wrong command line args"
	optName = sys.argv[1]
	optConfFile = sys.argv[2]
	clfName = sys.argv[3]
	clfConfigFile = sys.argv[4]
	
	feSelector = FeatureSelector(clfName, clfConfigFile)
	optimizer = createOptimizer(optName, optConfFile, feSelector)
	optimizer.run()
	print("best found")
	print(optimizer.getBest())

