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

import os
import sys
from random import randint
import time
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

if __name__ == "__main__":
	op = sys.argv[1]
	keyLen  = None

	classes = ["1", "0"]
	custType = ["1", "0"]

	if op == "generate":
		numSample = int(sys.argv[2])
		noise = float(sys.argv[3])
		if (len(sys.argv) == 5):
			keyLen = int(sys.argv[4])
	
		escalationDistr = CategoricalRejectSampler(("1", 30), ("0", 70))
		featCondDister = {}

		#num of days open
		key = ("1", 0)
		distr = NonParamRejectSampler(1, 1, 15, 20, 15, 30, 40, 55, 80, 120, 170)
		featCondDister[key] = distr
		key = ("0", 0)
		distr = NonParamRejectSampler(1, 1, 120, 90, 70, 30, 10)
		featCondDister[key] = distr

		#num of re open
		key = ("1", 1)
		distr = NonParamRejectSampler(0, 1, 20, 35, 80)
		featCondDister[key] = distr
		key = ("0", 1)
		distr = NonParamRejectSampler(0, 1, 120, 80, 20)
		featCondDister[key] = distr

		#num of messages
		key = ("1", 2)
		distr = GaussianRejectSampler(16, 2)
		featCondDister[key] = distr
		key = ("0", 2)
		distr = GaussianRejectSampler(4, 1)
		featCondDister[key] = distr

		#num of past tickets
		key = ("1", 3)
		distr = NonParamRejectSampler(0, 1, 15, 35, 70)
		featCondDister[key] = distr
		key = ("0", 3)
		distr = NonParamRejectSampler(0, 1, 120, 60)
		featCondDister[key] = distr

		#num of hours for first response
		key = ("1", 4)
		distr = GaussianRejectSampler(16, 4)
		featCondDister[key] = distr
		key = ("0", 4)
		distr = GaussianRejectSampler(8, 2)
		featCondDister[key] = distr

		#average time in hours for response message 
		key = ("1", 5)
		distr = GaussianRejectSampler(24, 2)
		featCondDister[key] = distr
		key = ("0", 5)
		distr = GaussianRejectSampler(12, 1)
		featCondDister[key] = distr

		#num of rea assignments
		key = ("1", 6)
		distr = NonParamRejectSampler(0, 1, 20, 40, 70)
		featCondDister[key] = distr
		key = ("0", 6)
		distr = NonParamRejectSampler(0, 1, 100, 40)
		featCondDister[key] = distr
	
		#customer type
		key = ("1", 7)
		distr = CategoricalRejectSampler(("0", 20), ("1", 80))
		featCondDister[key] = distr
		key = ("0", 7)
		distr = CategoricalRejectSampler(("0", 90), ("1", 10))
		featCondDister[key] = distr

		#error
		erDistr = GaussianRejectSampler(0, noise)
	
		#sampler
		sampler = AncestralSampler(escalationDistr, featCondDister, 8)

		for i in range(numSample):
			(claz, features) = sampler.sample()
		
			# add noise
			features[0] = int(addNoiseNum(features[0], erDistr))
			features[1] = int(addNoiseNum(features[1], erDistr))
			features[2] = int(addNoiseNum(features[2], erDistr))
			features[3] = int(addNoiseNum(features[3], erDistr))
			features[4] = int(addNoiseNum(features[4], erDistr))
			features[5] = int(addNoiseNum(features[5], erDistr))
			features[6] = int(addNoiseNum(features[6], erDistr))
			features[7] = addNoiseCat(features[7], custType, noise)

			claz = addNoiseCat(claz, classes, noise)

			strFeatures = list(map(lambda f: toStr(f, 3), features))
			rec =  ",".join(strFeatures) + "," + claz
			if keyLen:
				rec = genID(keyLen) + "," + rec
			print rec


