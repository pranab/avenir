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

sex = ["1", "0"]
married = ["1", "0"]
age = ["Y", "M", "O"]
income = ["L", "M", "H"]
smoker = ["NS", "SS", "SM"]
ethnicity = ["WH", "BL", "SA", "EA"]


if op == "generate":
	numSample = int(sys.argv[2])
	sexDistr = CategoricalRejectSampler(("1", 55), ("0", 45))
	marriedDistr = CategoricalRejectSampler(("1", 40), ("0", 60))
	ageDistr = CategoricalRejectSampler(("Y", 25), ("M", 35), ("O", 40))
	condIncDistr = dict()
	incomeDistr = CategoricalRejectSampler(("L", 75), ("M", 20), ("H", 5))
	condIncDistr["Y"] = incomeDistr
	incomeDistr = CategoricalRejectSampler(("L", 10), ("M", 80), ("H", 10))
	condIncDistr["M"] = incomeDistr
	incomeDistr = CategoricalRejectSampler(("L", 10), ("M", 20), ("H", 70))
	condIncDistr["O"] = incomeDistr
	ethDistr = CategoricalRejectSampler(("WH", 60), ("BL", 20), ("SA", 10), ("EA", 10))
	
	for i in range(numSample):
		sex = sexDistr.sample()
		married = marriedDistr.sample()
		age = ageDistr.sample()
		incomeDistr = condIncDistr[age]
		income = incomeDistr.sample()
		eth = ethDistr.sample()
	
		print "%s,%s,%s,%s,%s" %(sex,married,age,income,eth)

elif op == "genDummyVar":		
	file = sys.argv[2]
		
	catVars = {}
	catVars[2] = age
	catVars[3] = income
	catVars[4] = ethnicity
	rs = 5
	dummyVarGen = DummyVarGenerator(rs, catVars, "1", "0", ",")
	fp = open(file, "r")
	for row in fp:
		newRow = dummyVarGen.processRow(row.strip())
		print newRow.strip()
	fp.close()
