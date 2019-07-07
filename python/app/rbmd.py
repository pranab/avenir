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
from rbm import *

# classifier
rbm = RestrictedBoltzmanMachine(sys.argv[1])

# override config param
if len(sys.argv) > 2:
	#parameters over riiding config file
	for i in range(2, len(sys.argv)):
		items = sys.argv[i].split("=")
		rbm.setConfigParam(items[0], items[1])

# execute	
config = rbm.getConfig()	
mode = rbm.getMode()
print "running mode: " + mode
if mode == "train":
	rbm.train()

elif mode == "reconstruct":
	recon = rbm.reconstruct()
	for r in recon:
		print r

elif mode == "missing":
	data = rbm.getAnalyzeData()
	sh = data.shape
	nsamp = sh[0]
	counters = list(map(lambda i: dict(), range(nsamp)))
	itc = config.getIntConfig("analyze.recon.iter.count")[0]
	for i in range(itc):
		print "iteration ", i
		recons = rbm.reconstruct()
		for j in range(nsamp):
			recon = recons[j]
			inc = recon[5:8]
			print "sample ", str(j), " income ", str(inc)
			counter = counters[j]
			sinc = toStrFromList(inc, 3)
			incrKeyedCounter(counter, sinc)
	print "predicted missing values"
	for i in range(nsamp):
		counter = counters[i]
		rinc = max(counter, key=counter.get)	
		print "sample ", i, " income ", rinc
			