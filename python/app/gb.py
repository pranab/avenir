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
sys.path.append(os.path.abspath("../supv"))
from gbt import *

# classifier
gbtClass = GradientBoostedTrees(sys.argv[1])

# override config param
if len(sys.argv) > 2:
	#parameters over riiding config file
	for i in range(2, len(sys.argv)):
		items = sys.argv[i].split("=")
		gbtClass.setConfigParam(items[0], items[1])

# execute		
mode = gbtClass.getMode()
print "running mode: " + mode
if mode == "train":
	gbtClass.train()
elif mode == "trainValidate":
	if gbtClass.getSearchParamStrategy() is None:
		gbtClass.trainValidate()
	else:
		gbtClass.trainValidateSearch()
elif mode == "predict":
	clsData = gbtClass.predict()
	print clsData
elif mode == "validate":
	gbtClass.validate()
elif mode == "autoTrain":
	gbtClass.autoTrain()
else:
	print "invalid running mode" 

	
	
	