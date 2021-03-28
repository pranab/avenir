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
import statistics 
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../supv"))
from util import *
from mlutil import *
from sampler import *
from tnn import *

op = sys.argv[1]
keyLen  = None

classes = ["1", "0"]
sex = ["M", "F"]
smoker = ["NS", "SS", "SM"]
diet = ["BA", "AV", "GO"]
ethnicity = ["WH", "BL", "SA", "EA"]


def addLocalNoise(claz, noise, features):
	"""
	adds different noise to features partitions 
	"""
	lnoise = 0.2 * noise
	hnoise = 1.8 * noise
	n = randomFloat(lnoise, hnoise)
	
	
def findPart(rec, partitions):
	"""
	finds matching partition index
	"""
	matched = True
	mpi = None
	for (pi, part) in enumerate(partitions):
		#print(part)
		matched = True
		for i in range(0, len(part), 3):
			c = part[i]
			op = part[i+1]
			pval = part[i+2]
			dval = rec[c]
			#print("{},{},{},{}".format(c,op,pval,dval))
			if op == "LT":
				matched = matched and dval < pval
			elif op == "GE":
				matched = matched and dval >= pval
			elif op == "IN":
				pval = pval.split()
				matched = matched and dval in pval
			elif op == "NOTIN":
				pval = pval.split()
				matched = matched and dval not in pval
			if  not matched:
				break
		if matched:
			#print("matched")
			#print(part)
			#print(rec)
			mpi = pi
			break				
	
	assert	matched, "no matching partition found"
	return mpi
		
def createCatEncoder(keyLen):
	"""
	create categrival value encoder
	"""
	catVars = {}
	if keyLen:
		catVars[1] = sex
		catVars[6] = smoker
		catVars[7] = diet
		catVars[10] = ethnicity
		rs = 12
	else:
		catVars[0] = sex
		catVars[5] = smoker
		catVars[6] = diet
		catVars[9] = ethnicity
		rs = 11
	dummyVarGen = DummyVarGenerator(rs, catVars, "1", "0", ",")
	return dummyVarGen

def encodeCat(encoder, rec):
	"""
	encode cat fields
	"""
	srec = list(map(lambda v : toStr(v,3), rec))
	srec = ",".join(srec)
	srec = encoder.processRow(srec)
	return srec.split(",")
	

def showAccuracies(scores):
	"""
	displays accuracies
	"""
	print("scores")
	print(scores)
	mscore = statistics.mean(scores)
	sdscore = statistics.stdev(scores, xbar=mscore)
	print("accuracy mean {:.3f}  std dev {:.3f}".format(mscore, sdscore))
	drawHist(scores, "Accuracy distr", "accuracy", "frequency")


if op == "generate":
	"""
	generate data
	"""
	numSample = int(sys.argv[2])
	noise = float(sys.argv[3])
	if (len(sys.argv) == 5):
		keyLen = int(sys.argv[4])
	
	diseaseDistr = CategoricalRejectSampler(("1", 25), ("0", 75))
	featCondDister = {}
	
	#sex
	key = ("1", 0)
	distr = CategoricalRejectSampler(("M", 60), ("F", 40))
	featCondDister[key] = distr
	key = ("0", 0)
	distr = CategoricalRejectSampler(("M", 50), ("F", 50))
	featCondDister[key] = distr
	
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
	
	#diet
	key = ("1", 6)
	distr = CategoricalRejectSampler(("BA", 60), ("AV", 35), ("GO", 20))
	featCondDister[key] = distr
	key = ("0", 6)
	distr = CategoricalRejectSampler(("BA", 15), ("AV", 40), ("GO", 45))
	featCondDister[key] = distr

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
	
	#error
	sampler = AncestralSampler(diseaseDistr, featCondDister, 10)
	if noise > .001:
		erDistr = GaussianRejectSampler(0, noise)

	for i in range(numSample):
		(claz, features) = sampler.sample()
		
		# add noise
		if noise > .001:
			features[2] = int(addNoiseNum(features[2], erDistr))
			features[7] = int(addNoiseNum(features[7], erDistr))
			features[8] = int(addNoiseNum(features[8], erDistr))
			features[0] = addNoiseCat(features[0], sex, noise)
			features[5] = addNoiseCat(features[5], smoker, noise)
			features[6] = addNoiseCat(features[6], diet, noise)
			features[9] = addNoiseCat(features[9], ethnicity, noise)
		
			claz = addNoiseCat(claz, classes, noise)
		else:
			features[2] = int(features[2])
			features[7] = int(features[7])
			features[8] = int(features[8])
		
		strFeatures = [toStr(f, 3) for f in features]
		rec =  ",".join(strFeatures) + "," + claz
		if keyLen:
			rec = genID(keyLen) + "," + rec
		print(rec)
		
elif op == "genDummyVar":		
	"""
	encode categorical data
	"""
	file = sys.argv[2]
	if (len(sys.argv) == 4):
		keyLen = int(sys.argv[3])
		
	catVars = {}
	if keyLen:
		catVars[1] = sex
		catVars[6] = smoker
		catVars[7] = diet
		catVars[10] = ethnicity
		rs = 12
	else:
		catVars[0] = sex
		catVars[5] = smoker
		catVars[6] = diet
		catVars[9] = ethnicity
		rs = 11
	dummyVarGen = DummyVarGenerator(rs, catVars, "1", "0", ",")
	fp = open(file, "r")
	for row in fp:
		newRow = dummyVarGen.processRow(row)
		print(newRow.strip())
	fp.close()


elif op == "addNoise":
	"""
	add additional noise
	"""
	filePath = sys.argv[2]
	for rec in fileRecGen(filePath, ","):
		rec[8] = str(int(addNoiseNum(int(rec[8]), UniformNumericSampler(-0.4, 0.4))))
		rec[9] = str(int(addNoiseNum(int(rec[9]), UniformNumericSampler(-0.5, 0.5))))
		print(",".join(rec))
				
elif op == "adLocdNoise":
	"""
	add local partition based noise
	"""
	filePath = sys.argv[2]
	noise = float(sys.argv[3])
	nlo = 0.2 * noise
	nhi = 1.8 * noise
	types = "0:string,1:cat:M F,2:int,3:int,4:int,5:int,6:cat:NS SS SM,7:cat:BA AV GO,8:int,9:int,10:cat:WH BL SA EA,11:int"
	tdata = getFileAsTypedRecords(filePath, types)
	columns = [1,2,3,6]
	partitions = getDataPartitions(tdata, types, columns)
	nlevels = list(map(lambda i : randomFloat(nlo, nhi), range(len(partitions))))
	#exitWithMsg("testing")
	
	for rec in tdata:
		pi = findPart(rec, partitions)
		nlevel = nlevels[pi]
		claz = str(rec[len(rec)-1])
		clazb = claz
		claz = addNoiseCat(claz, classes, nlevel)	
		srec = list(map(lambda v : toStr(v,3), rec))
		#print("{:.3f} {} {}, ".format(nlevel,clazb,claz))
		srec[len(srec)-1] = claz
		print(",".join(srec))
		

elif op == "spuriousFeatures":
	"""
	add irrelevant features
	"""
	filePath = sys.argv[2]
	numYearsInJobDistr  = GaussianRejectSampler(8, 3)
	incomeDistr = GaussianRejectSampler(100, 40)
	for rec in fileRecGen(filePath, ","):
		newRec = rec[:-1]
		clAttr = int(rec[-1])
	
		#num of years in job
		numYr = int(numYearsInJobDistr.sample())
		numYr = 0 if numYr < 0 else numYr
		newRec.append(str(numYr))
	
		#income
		income = int(incomeDistr.sample())
		income = 20 if income < 20 else income
		newRec.append(str(income))
	
		newRec.append(rec[-1])
		print(",".join(newRec))

elif op == "nnTrain":
	"""
	tran neural network model
	"""
	prFile = sys.argv[2]
	clflier = FeedForwardNetwork(prFile)
	clflier.buildModel()
	FeedForwardNetwork.batchTrain(clflier)

elif op == "nnAccuracyByPartition":
	"""
	accuracies for partitioned data
	"""
	prFile = sys.argv[2]
	filePath = sys.argv[3]
	keyLen = int(sys.argv[4])
	types = "0:string,1:cat:M F,2:int,3:int,4:int,5:int,6:cat:NS SS SM,7:cat:BA AV GO,8:int,9:int,10:cat:WH BL SA EA,11:int"
	tdata = getFileAsTypedRecords(filePath, types)
	columns = columns = [1,2,3,6,7,8]
	partitions = getDataPartitions(tdata, types, columns)
	pdata = list(map(lambda i : list(), range(len(partitions))))
	
	encoder = createCatEncoder(keyLen)
	
	#create data partitions
	for rec in tdata:
		pi = findPart(rec, partitions)
		
		#encode categorical values
		srec = list(map(lambda v : toStr(v,3), rec))
		srec = ",".join(srec)
		srec = encoder.processRow(srec)
		pdata[pi].append(srec.split(","))
		
	#performance score for each data partition	
	clflier = FeedForwardNetwork(prFile)
	clflier.buildModel()
	scores = list()
	clflier.setVerbose(False)
	for pd in pdata:
		print("partition size {}".format(len(pd)))
		if len(pd) > 10:
			FeedForwardNetwork.prepValidate(clflier, pd)
			score = FeedForwardNetwork.validateModel(clflier)
			scores.append(score)
		
	print(scores)
	mscore = statistics.mean(scores)
	sdscore = statistics.stdev(scores, xbar=mscore)
	print("accuracy mean {:.3f}  std dev {:.3f}".format(mscore, sdscore))
	drawHist(scores, "Accuracy distr", "accuracy", "frequency")

elif op == "nnAccuracyByShift":
	"""
	acciracies for shifted data
	"""
	prFile = sys.argv[2]
	filePath = sys.argv[3]
	keyLen = int(sys.argv[4])
	nshift = int(sys.argv[5])
	maxShift = float(sys.argv[6])
	maxScale = float(sys.argv[7])
	types = "0:string,1:cat:M F,2:int,3:int,4:int,5:int,6:cat:NS SS SM,7:cat:BA AV GO,8:int,9:int,10:cat:WH BL SA EA,11:int"
	tdata = getFileAsTypedRecords(filePath, types)
	ftypes = "0:string,1:cat:M F,2:int,3:int,4:int,5:int,6:cat:NS SS SM,7:cat:BA AV GO,8:int,9:int,10:cat:WH BL SA EA"
	dgen = ShiftedDataGenerator(ftypes, tdata, maxShift, maxScale)
	
	clflier = FeedForwardNetwork(prFile)
	clflier.buildModel()
	scores = list()
	clflier.setVerbose(False)
	encoder = createCatEncoder(keyLen)

	for _ in range(nshift):
		ttdata = dgen.transform(tdata)
		ettdata = list(map(lambda r : encodeCat(encoder, r), ttdata))
			
		FeedForwardNetwork.prepValidate(clflier, ettdata)
		score = FeedForwardNetwork.validateModel(clflier)
		scores.append(score)
		
	showAccuracies(scores)	
	
else:
	exitWithMsg("inbvalid command")	
	
