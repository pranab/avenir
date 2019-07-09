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
from rbm import *

def missingValueBySampling(rbm, config):
	data = rbm.getAnalyzeData()
	sh = data.shape
	nsamp = sh[0]
	counters = list(map(lambda i: dict(), range(nsamp)))
	
	# random in initial value
	mBeg = config.getIntConfig("analyze.missing.beg.col")[0]
	mEnd = config.getIntConfig("analyze.missing.end.col")[0]
	size = mEnd - mBeg
	assert size > 0, "invalid missing columns"
	invc = config.getIntConfig("analyze.missing.initial.count")[0]
	itc = config.getIntConfig("analyze.missing.iter.count")[0]
	allInitial = config.getBooleanConfig("analyze.missing.all.initial")[0]
	if allInitial:
		invc = size if size > 1 else 2
	bValues = [[0], [1]]
	for inv in range(invc):
		#rset initial
		indx = inv if allInitial else -1
		for d in data:
			if size > 1:
				onh = createOneHotVec(size, indx)
			else:
				if indx >= 0:
					onh = bValues[inv]
				else:
					onh = selectRandomFromList(bValues)
			for i in range(size):
				d[mBeg+i] = onh[i]	
		
		# multiple gibbs sampling	
		for i in range(itc):
			print "iteration ", i
			recons = rbm.reconstruct()
			for j in range(nsamp):
				recon = recons[j]
				inc = recon[mBeg:mEnd]
				print "sample ", j, " income ", str(inc)
				counter = counters[j]
				sinc = toStrFromList(inc, 3)
				incrKeyedCounter(counter, sinc)
	
		print "predicted missing values"
		for i in range(nsamp):
			counter = counters[i]
			rinc = max(counter, key=counter.get)	
			print "sample ", i, " income ", rinc
	
	
	validate = config.getBooleanConfig("analyze.missing.validate")[0]
	if validate:
		predicted = list(map(lambda counter: max(counter, key=counter.get), counters))
		
		vaFilepath = config.getStringConfig("analyze.missing.validate.file.path")[0]
		assert vaFilepath, "missing missing value validation file path"
		actual = list()
		for rec in fileRecGen(vaFilepath, ","):
			inc = rec[mBeg:mEnd]
			sinc = ",".join(inc)
			print "actual ", sinc
			actual.append(sinc)
		assert (len(predicted) == len(actual)), "un equal size of predicted and validation"
		zipped = zip(predicted, actual)
		corCount = len(list(filter(lambda z: z[0] == z[1], zipped)))
		accuracy = (corCount * 100.0) / nsamp
		print "accuracy %.3f" %(accuracy)


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
	missingValueBySampling(rbm, config)
		 
			