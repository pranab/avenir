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
from rf import *
from interpret import *

# classifier

rfClass = RandomForest(sys.argv[1])

def processParams():
	# override config param
	if len(sys.argv) > 2:
		#parameters over riiding config file
		for i in range(2, len(sys.argv)):
			items = sys.argv[i].split("=")
			rfClass.setConfigParam(items[0], items[1])

# execute		
mode = sys.argv[2]
print "running mode: " + mode
if mode == "train":
	rfClass.train()
elif mode == "trainValidate":
	if rfClass.getSearchParamStrategy() is None:
		rfClass.trainValidate()
	else:
		rfClass.trainValidateSearch()
elif mode == "predict":
	clsData = rfClass.predict()
	print clsData
elif mode == "validate":
	rfClass.validate()
elif mode == "explain":
	intr = LimeInterpreter(sys.argv[3])
	rec = sys.argv[4]
	featData = rfClass.prepTrainingData()[0]
else:
	print "invalid running mode " + mode 

	
	
	
