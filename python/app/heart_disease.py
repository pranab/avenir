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

op = sys.argv[1]
keyLen  = None
if (len(sys.argv) == 4):
	keyLen = int(sys.argv[3])

if op == "generate":
	numSample = int(sys.argv[2])
	diseaseDistr = CategoricalRejectSampler(("1", 25), ("0", 75))
	featCondDister = {}
	classes = ["1", "0"]
	
	#sex
	key = ("1", 0)
	distr = CategoricalRejectSampler(("M", 60), ("F", 40))
	featCondDister[key] = distr
	key = ("0", 0)
	distr = CategoricalRejectSampler(("M", 50), ("F", 50))
	featCondDister[key] = distr
	sex = ["M", "F"]
	
	#age
	key = ("1", 1)
	distr = NonParamRejectSampler(30, 10, 10, 20, 35, 60, 90)
	featCondDister[key] = distr
	key = ("0", 1)
	distr = NonParamRejectSampler(30, 10, 15, 20, 25, 30, 30)
	featCondDister[key] = distr

	#weight
	key = ("1", 2)
	distr = GaussianRejectSampler(190, 8)
	featCondDister[key] = distr
	key = ("0", 2)
	distr = GaussianRejectSampler(150, 15)
	featCondDister[key] = distr

	#systolic blood pressure
	key = ("1", 3)
	distr = NonParamRejectSampler(100, 10, 20, 25, 25, 30, 35, 45, 60, 75)
	featCondDister[key] = distr
	key = ("0", 3)
	distr = NonParamRejectSampler(100, 10, 20, 30, 40, 20, 12, 8, 6, 4)
	featCondDister[key] = distr

	#dialstolic  blood pressure
	key = ("1", 4)
	distr = NonParamRejectSampler(60, 10, 20, 20, 25, 35, 50, 70)
	featCondDister[key] = distr
	key = ("0", 4)
	distr = NonParamRejectSampler(60, 10, 20, 20, 25, 18, 12, 7)
	featCondDister[key] = distr

	#smoker
	key = ("1", 5)
	distr = CategoricalRejectSampler(("NS", 20), ("SS", 35), ("SM", 60))
	featCondDister[key] = distr
	key = ("0", 5)
	distr = CategoricalRejectSampler(("NS", 40), ("SS", 20), ("SM", 15))
	featCondDister[key] = distr
	smoker = ["NS", "SS", "SM"]
	
	#diet
	key = ("1", 6)
	distr = CategoricalRejectSampler(("BA", 60), ("AV", 35), ("GO", 20))
	featCondDister[key] = distr
	key = ("0", 6)
	distr = CategoricalRejectSampler(("BA", 15), ("AV", 40), ("GO", 45))
	featCondDister[key] = distr
	diet = ["BA", "AV", "GO"]

	#physical activity per week
	key = ("1", 7)
	distr = GaussianRejectSampler(5, 1)
	featCondDister[key] = distr
	key = ("0", 7)
	distr = GaussianRejectSampler(15, 2)
	featCondDister[key] = distr

	#education
	key = ("1", 8)
	distr = GaussianRejectSampler(11, 2)
	featCondDister[key] = distr
	key = ("0", 8)
	distr = GaussianRejectSampler(17, 1)
	featCondDister[key] = distr

	#ethnicity
	key = ("1", 9)
	distr = CategoricalRejectSampler(("WH", 30), ("BL", 40), ("SA", 50), ("EA", 20))
	featCondDister[key] = distr
	key = ("0", 9)
	distr = CategoricalRejectSampler(("WH", 50), ("BL", 20), ("SA", 16), ("EA", 20))
	featCondDister[key] = distr
	ethnicity = ["WH", "BL", "SA", "EA"]
	
	#error
	erDistr = GaussianRejectSampler(0, .05)
	
	sampler = AncestralSampler(diseaseDistr, featCondDister, 10)

	for i in range(numSample):
		(claz, features) = sampler.sample()
		
		# add noise
		features[2] = int(addNoiseNum(features[2], erDistr))
		features[7] = int(addNoiseNum(features[7], erDistr))
		features[8] = int(addNoiseNum(features[8], erDistr))
		
		features[0] = addNoiseCat(features[0], sex, .05)
		features[5] = addNoiseCat(features[5], smoker, .05)
		features[6] = addNoiseCat(features[6], diet, .05)
		features[9] = addNoiseCat(features[9], ethnicity, .05)
		
		claz = addNoiseCat(claz, classes, .08)
		
		strFeatures = [toStr(f, 3) for f in features]
		rec =  ",".join(strFeatures) + "," + claz
		if keyLen:
			rec = genID(keyLen) + "," + rec
		print rec
		
elif op == "genDummyVar":		
		file = sys.argv[2]
		catVars = {}
		if keyLen:
			catVars[1] = ("M", "F")
			catVars[6] = ("NS", "SS", "SM")
			catVars[7] = ("BA", "AV", "GO")
			catVars[10] = ("WH", "BL", "SA", "EA")
			rs = 12
		else:
			catVars[0] = ("M", "F")
			catVars[5] = ("NS", "SS", "SM")
			catVars[6] = ("BA", "AV", "GO")
			catVars[9] = ("WH", "BL", "SA", "EA")
			rs = 11
		dummyVarGen = DummyVarGenerator(rs, catVars, "1", "0", ",")
		fp = open(file, "r")
		for row in fp:
			newRow = dummyVarGen.processRow(row)
			print newRow.strip()
		fp.close()
	
	
