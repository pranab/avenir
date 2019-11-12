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
sys.path.append(os.path.abspath("../supv"))
sys.path.append(os.path.abspath("../mlextra"))
import interpret
from rf import *
from svm import *
from gbt import *
from interpret import *

#print '\n'.join(sys.path)
#print interpret.__file__

# classifier
if __name__ == "__main__":
	i = 1
	mode = sys.argv[i]
	i += 1

	clfName = sys.argv[i]
	i += 1

	clfConfigFile = sys.argv[i]
	i += 1
	if clfName == "rf":
		clf = RandomForest(clfConfigFile)
	elif clfName == "gbt":
		clf = GradientBoostedTrees(clfConfigFile)
	elif clfName == "svm":
		clf = SupportVectorMachine(clfConfigFile)
	else:
		raise valueError("unsupported classifier")
	

	predFun = lambda x: clf.predictProb(x)


	# execute		
	verbose = clf.getConfig().getBooleanConfig("common.verbose")[0]
	print ("running mode: " + mode)
	if mode == "train":
		clf.train()
	elif mode == "trainValidate":
		if clf.getSearchParamStrategy() is None:
			clf.trainValidate()
		else:
			clf.trainValidateSearch()
	elif mode == "predict":
		clsData = clf.predict()
		print (clsData)
	elif mode == "validate":
		clf.validate()
	elif mode == "explain":
		intr = LimeInterpreter(sys.argv[i])
		i += 1
		rec = sys.argv[i]
		i += 1

		rec = clf.prepStringPredictData(rec.decode('utf-8'))
		featData = clf.prepTrainingData()[0]
		if verbose:
			print ("feature shape ",featData.shape)
		intr.buildExplainer(featData)
		exp = intr.explain(rec, predFun)
		print ("model explanation")
		print(exp.as_list())
		#fig = exp.as_pyplot_figure()
		#fig.show()
	else:
		print ("invalid running mode " + mode)

	
	
	
