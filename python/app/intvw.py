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

import os
import sys
from random import randint
import time
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
from util import *
from sampler import *
from hbias import *

"""
candidate phone interview
"""
if __name__ == "__main__":

	op = sys.argv[1]
	classes = ["T", "F"]
	sex = ["M", "F"]
	education = ["H", "B", "M"]


	if op == "gen":
		numSample = int(sys.argv[2])
		noise = float(sys.argv[3])
	
		selDistr = CategoricalRejectSampler(("T", 30), ("F", 70))
		featCondDister = {}

		#sex
		key = ("T", 0)
		distr = CategoricalRejectSampler(("M", 60), ("F", 40))
		featCondDister[key] = distr
		key = ("F", 0)
		distr = CategoricalRejectSampler(("M", 60), ("F", 40))
		featCondDister[key] = distr
		
		#education
		key = ("T", 1)
		distr = CategoricalRejectSampler(("H", 20), ("B", 40), ("M", 30))
		featCondDister[key] = distr
		key = ("F", 1)
		distr = CategoricalRejectSampler(("H", 30), ("B", 20), ("M", 10))
		featCondDister[key] = distr
		
		#experience
		key = ("T", 2)
		distr = NonParamRejectSampler(1, 1, 10, 20, 35, 60, 60)
		featCondDister[key] = distr
		key = ("F", 2)
		distr = NonParamRejectSampler(1, 1, 25, 30, 15, 10, 5)
		featCondDister[key] = distr
		
		#rmployment gap
		key = ("T", 3)
		distr = NormalSampler(2.0, 0.8)
		featCondDister[key] = distr
		key = ("F", 3)
		distr = NormalSampler(6.0, 1.6)
		featCondDister[key] = distr

		#phone interview
		key = ("T", 4)
		distr = NonParamRejectSampler(1, 1, 10, 20, 30, 40, 45)
		featCondDister[key] = distr
		key = ("F", 4)
		distr = NonParamRejectSampler(1, 1, 35, 25, 15, 10, 5)
		featCondDister[key] = distr

		sampler = AncestralSampler(selDistr, featCondDister, 5)

		for i in range(numSample):
			(claz, features) = sampler.sample()
			gap = int(features[3])
			gap = 0 if gap < 0 else gap
			features[3] = gap
			claz = addNoiseCat(claz, classes, noise)
			strFeatures = list(map(lambda f : toStr(f, 3), features))
			rec =  genID(10) + "," + ",".join(strFeatures) + "," + claz
			print(rec)

	elif op == "bias":
		fpath = sys.argv[2]
		bias = int(sys.argv[3])
		for rec in fileRecGen(fpath):
			if isEventSampled(bias):
				if rec[1] == "M":
					rec[6] = "T"
			print(",".join(rec))

	elif op == "elift":
		fpath = sys.argv[2]
		ftypes = [3, "int", 4, "int", 5, "int"]
		bd = BiasDetector(fpath, ftypes)
		fe = [2, "B"]
		pfe = [1, "M"]
		cl = [6, "T"]
		re = bd.extLift(fe, pfe, cl)
		printMap(re, "item", "value", 3, offset=24)
		
	else:
		exitWithMsg("invalid command")
		
		