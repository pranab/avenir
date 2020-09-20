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
import random
import statistics 
import matplotlib.pyplot as plt
import pprint
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
from util import *
from sampler import *
from mcsim import *
from daexp import *

"""
generates statistics of 2 sample statistic with MC simulation
"""


def genStat(args):
	"""
	
	"""
	distr1 = args[:10]
	i = 10
	expl = args[i]
	i += 1
	twoSampStat = args[i]
	i += 1
	nsamp = args[i]
	
	#generate second distr by mutating first
	nmut = randomInt(0, 5)
	distr2 = distr1.copy()
	mutateList(distr2, nmut, 10.0, 100.0)
	
	sampler1 = NonParamRejectSampler(0, 10, distr1)
	sampler1.sampleAsFloat()
	sampler2 = NonParamRejectSampler(0, 10, distr2)
	sampler2.sampleAsFloat()
	
	
	sample1 = randomSampledList(sampler1, nsamp)
	sample2 = randomSampledList(sampler2, nsamp)
	expl.addListNumericData(sample1, "ds1")
	expl.addListNumericData(sample2, "ds2")
	
	if twoSampStat == "ks":
		res = expl.testTwoSampleKs(self, "ds1", "ds2")
	elif twoSampStat == "and":
		res = expl.testTwoSampleAnderson("ds1", "ds2")
	elif twoSampStat == "cvm":
		res = expl.testTwoSampleCvm("ds1", "ds2")
	else:
		raise ValueError("invalid 2 sample statistic")
		
	stat = res["stat"]
	return stat
	

if __name__ == "__main__":
	assert len(sys.argv) == 4, "ivalid number of command line args, expecting 3"
	numIter = int(sys.argv[1])
	twoSampStat = sys.argv[2]
	nsamp = int(sys.argv[3])
	simulator = MonteCarloSimulator(numIter, genStat, "./log/mcsim.log", "info")
	for _ in range(10):
		simulator.registerUniformSampler(10.0, 100.0)
	expl = DataExplorer(False)
	simulator.registerExtraArgs(expl, twoSampStat, nsamp)
	simulator.run()
	
	simulator.drawHist(twoSampStat + "2 sample stat", "stat", "distr")
	print("mean {:.3f}  sd {:.3f}  min {:.3f}".format(simulator.getMean(), simulator.getStdDev(), simulator.getMin()))
	cvalues = simulator.getLowerTailStat(0.5)
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(cvalues)



