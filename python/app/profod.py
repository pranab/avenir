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
sys.path.append(os.path.abspath("../unsupv"))
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from profo import *

if __name__ == "__main__":
	# classifier
	forecaster = ProphetForcaster(sys.argv[1], None, None)

	# override config param
	if len(sys.argv) > 2:
		#parameters over riiding config file
		for i in range(2, len(sys.argv)):
			items = sys.argv[i].split("=")
			forecaster.setConfigParam(items[0], items[1])

	# execute	
	config = forecaster.getConfig()	
	mode = forecaster.getMode()
	print "running mode: " + mode
	if mode == "train":
		forecaster.train()
	elif mode == "forecast":
		forecaster.forecast()
	elif mode == "validate":
		forecaster.validate()
	elif mode == "shuffle":
		forecaster.shuffle()



