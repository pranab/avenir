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
from array import *
import statistics
import numpy as np
from sklearn.neighbors import KDTree
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../supv"))
sys.path.append(os.path.abspath("../mlextra"))

from mlutil import *
from util import *
from sampler import *
from tnn import *
from mcalib import *
from optpopu import *
from daexp import *


NFEAT = 11
NFEAT_EXT = 14

class LoanCounterFacCost(object):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, baseData, fieldMap):
		"""
		intialize
		"""
		self.logger = None

	def isValid(self, args):
		"""
		candidate validation
		"""
		return True
		
	def evaluate(self, args):
		"""
		cost
		"""
		cost = 0
		
		return cost
			

class LoanApprove:
	def __init__(self, numLoans=None):
		self.numLoans = numLoans
		self.marStatus = ["married", "single", "divorced"]
		self.loanTerm = ["7", "15", "30"]
		self.addExtra = False
		
	def initOne(self):
		"""
		initialize samplers
		"""
		self.threshold = 118
		self.margin = 5

		# distributions
		self.marriedDistr = CategoricalRejectSampler(("married", 80), ("single", 100), ("divorced", 30))
		self.numChildDistr = CategoricalRejectSampler(("1", 80), ("2", 100), ("2", 40))
		self.eduDistr = CategoricalRejectSampler(("1", 60), ("2", 100), ("3", 30))
		self.selfEmployedDistr = CategoricalRejectSampler(("1", 30), ("0", 100))
		self.incomeDistr = GaussianRejectSampler(100,20)
		self.numYearsExpDistr = GaussianRejectSampler(10,3)
		self.outstandingLoanDistr = GaussianRejectSampler(20,5)
		self.loanAmDistr = GaussianRejectSampler(300,70)
		self.loanTermDistr = CategoricalRejectSampler(("10", 40), ("15", 60), ("30", 100))
		self.credScoreDistr = GaussianRejectSampler(700,50)
		zipClusterDistr = [("high", 30), ("average", 100), ("low", 60)]
		zipClusters = {\
		"high" : ["95061", "95062", "95064", "95065", "95067"], \
		"average" : ["95103", "95104", "95106", "95107", "95109", "95113", "95115", "95118", "95121" ], \
		"low" : ["95376", "95377", "95378", "95353", "95354", "95356"]}
		self.zipDistr = ClusterSampler(zipClusters, ("high", 30), ("average", 100), ("low", 60))
		
		# scores
		self.marriedScore = {"married" : 16, "single" : 10, "divorced" : 6}
		self.numChildScore = {1 : 12, 2 : 9 , 3 : 4}
		self.eduScore = {1 : 7 , 2 : 12, 3 : 15}
		self.selfEmployedScore = {"0" : 15, "1" : 11}
		self.incomeScore = StepFunction((50, 70, 2), (70, 90, 5), (90, 100, 8), (100, 110, 12),\
		(110, 130, 14), (130, 150, 18))
		self.numYearsExpScore = StepFunction((6, 10, 4), (10, 14, 9), (14, 20, 13))
		self.outstandingLoanScore = StepFunction((2, 4, 16), (4, 8, 13), (8, 14, 10), (14, 22, 8),\
		(22, 32, 6), (32, 44, 2))
		self.loanAmScore = StepFunction((200, 250, 22), (250, 300, 20), (300, 350, 16), (350, 400, 10),\
		(400, 450, 5), (450, 500, 2))
		self.loanTermScore = {10 : 15, 15 : 18 , 30 : 23}
		self.credScoreScore = StepFunction((600, 650, 8), (650, 700, 12), (700, 750, 17), (750, 800, 23),\
		(800, 850, 31))
		self.zipRateScore = {"high" : 17, "average" : 15, "low" : 11}

	def generateOne(self):
		"""
		sample
		"""
		self.initOne()
		posCount = 0
		for i in range(self.numLoans):
			id = genID(10)
			married = self.marriedDistr.sample()
			numChild = int(self.numChildDistr.sample())
			edu = int(self.eduDistr.sample())
			selfEmployed = self.selfEmployedDistr.sample()
			income = int(self.incomeDistr.sample())
			income = rangeSample(income, 50, 160)
			numYearsExp = int(self.numYearsExpDistr.sample())
			numYearsExp = rangeSample(numYearsExp, 6, 20)
			outstandingLoan = int(self.outstandingLoanDistr.sample())
			loanAm = int(self.loanAmDistr.sample())
			loanAm = rangeSample(loanAm, 200, 500)
			loanTerm  = int(self.loanTermDistr.sample())
			credScore = int(self.credScoreDistr.sample())
			credScore = rangeSample(credScore, 600, 850)
			(zipRate, zipCode)  = self.zipDistr.sample()

			# score for each score
			score = 0
			score +=  self.marriedScore[married]
			score +=  self.numChildScore[numChild]
			score +=  self.eduScore[edu]
			score +=  self.selfEmployedScore[selfEmployed]
			score +=  self.incomeScore.find(income)
			score +=  self.numYearsExpScore.find(numYearsExp)
			score +=  self.outstandingLoanScore.find(outstandingLoan)
			score +=  self.loanTermScore[loanTerm]
			score +=  self.credScoreScore.find(credScore)
			score +=  self.zipRateScore[zipRate]

			# feature coupling
			if (income > 140 and loanAm < 300):
				score += 10
			if (income < 80 and loanAm > 280):
				score -= 12
			
			if (credScore > 760 and loanAm < 320):
				score += 12
			if (credScore < 700 and loanAm > 260):
				score -= 14

			if (numChild == 3 and income < 100):
				score -= 8

			# outcome
			if score > (self.threshold + self.margin):
				approved = 1
			elif score < (self.threshold - self.margin):
				approved = 0
			else:
				if randint(0, 100) < 50:
					approved = 1
				else:
					approved = 0

			if approved == 1:
				posCount += 1

			print ("{},{},{},{},{},{},{},{},{},{},{},{},{}".format(id, married, numChild, edu, selfEmployed, income,\
			numYearsExp, outstandingLoan, loanAm, loanTerm, credScore, zipCode, approved))

		#print "positive count " + str(posCount)
		
	def initTwo(self):
		"""
		initialize samplers
		"""
		self.approvDistr = CategoricalRejectSampler(("1", 60), ("0", 40))
		self.featCondDister = {}
		
		#marital status
		key = ("1", 0)
		distr = CategoricalRejectSampler(("married", 100), ("single", 60), ("divorced", 40))
		self.featCondDister[key] = distr
		key = ("0", 0)
		distr = CategoricalRejectSampler(("married", 40), ("single", 100), ("divorced", 40))
		self.featCondDister[key] = distr
	
		
		# num of children
		key = ("1", 1)
		distr = CategoricalRejectSampler(("1", 100), ("2", 90), ("3", 40))
		self.featCondDister[key] = distr
		key = ("0", 1)
		distr = CategoricalRejectSampler(("1", 50), ("2", 70), ("3", 100))
		self.featCondDister[key] = distr

		# education
		key = ("1", 2)
		distr = CategoricalRejectSampler(("1", 30), ("2", 80), ("3", 100))
		self.featCondDister[key] = distr
		key = ("0", 2)
		distr = CategoricalRejectSampler(("1", 100), ("2", 40), ("3", 30))
		self.featCondDister[key] = distr

		#self employed
		key = ("1", 3)
		distr = CategoricalRejectSampler(("1", 40), ("0", 100))
		self.featCondDister[key] = distr
		key = ("0", 3)
		distr = CategoricalRejectSampler(("1", 100), ("0", 30))
		self.featCondDister[key] = distr
		
		# income
		key = ("1", 4)
		distr = GaussianRejectSampler(120,15)
		self.featCondDister[key] = distr
		key = ("0", 4)
		distr = GaussianRejectSampler(50,10)
		self.featCondDister[key] = distr

		# years of experience
		key = ("1", 5)
		distr = GaussianRejectSampler(15,3)
		self.featCondDister[key] = distr
		key = ("0", 5)
		distr = GaussianRejectSampler(5,1)
		self.featCondDister[key] = distr

		# number of years in current job
		key = ("1", 6)
		distr = GaussianRejectSampler(3,.5)
		self.featCondDister[key] = distr
		key = ("0", 6)
		distr = GaussianRejectSampler(1,.2)
		self.featCondDister[key] = distr

		# outstanding debt
		key = ("1", 7)
		distr = GaussianRejectSampler(20,5)
		self.featCondDister[key] = distr
		key = ("0", 7)
		distr = GaussianRejectSampler(60,10)
		self.featCondDister[key] = distr
		
		# loan amount
		key = ("1", 8)
		distr = GaussianRejectSampler(300,50)
		self.featCondDister[key] = distr
		key = ("0", 8)
		distr = GaussianRejectSampler(600,50)
		self.featCondDister[key] = distr
		
		# loan term 
		key = ("1", 9)
		distr = CategoricalRejectSampler(("7", 100), ("15", 40), ("30", 60))
		self.featCondDister[key] = distr
		key = ("0", 9)
		distr = CategoricalRejectSampler(("7", 30), ("15", 100), ("30", 60))
		self.featCondDister[key] = distr
		
		# credit score
		key = ("1", 10)
		distr = GaussianRejectSampler(700,20)
		self.featCondDister[key] = distr
		key = ("0", 10)
		distr = GaussianRejectSampler(500,50)
		self.featCondDister[key] = distr
		
		if self.addExtra:
			# saving
			key = ("1", 11)
			distr = NormalSampler(80,10)
			self.featCondDister[key] = distr
			key = ("0", 11)
			distr = NormalSampler(60,8)
			self.featCondDister[key] = distr
			
			# retirement
			zDistr = NormalSampler(0, 0)
			key = ("1", 12)
			sDistr = DiscreteRejectSampler(0,1,1,20,80)
			nzDistr = NormalSampler(100,20)
			distr = DistrMixtureSampler(sDistr, zDistr, nzDistr)
			self.featCondDister[key] = distr
			key = ("0", 12)
			sDistr = DiscreteRejectSampler(0,1,1,50,50)
			nzDistr = NormalSampler(40,10)
			distr = DistrMixtureSampler(sDistr, zDistr, nzDistr)
			self.featCondDister[key] = distr
		
			#num of prior mortgae loans
			key = ("1", 13)
			distr = DiscreteRejectSampler(0,3,1,20,60,40,15)
			self.featCondDister[key] = distr
			key = ("0", 13)
			distr = DiscreteRejectSampler(0,1,1,70,30)
			self.featCondDister[key] = distr
			
		
	def generateTwo(self, noise, keyLen, addExtra):
		"""
		ancestral sampling
		"""
		self.addExtra = addExtra
		self.initTwo()
		
		#error
		erDistr = GaussianRejectSampler(0, noise)
	
		#sampler
		numChildren = NFEAT_EXT if self.addExtra else NFEAT
		sampler = AncestralSampler(self.approvDistr, self.featCondDister, numChildren)

		for i in range(self.numLoans):
			(claz, features) = sampler.sample()
		
			# add noise
			features[4] = int(features[4])
			features[7] = int(features[7])
			features[8] = int(features[8])
			features[10] = int(features[10])
			if self.addExtra:
				features[11] = int(features[11])
				features[12] = int(features[12])

			claz = addNoiseCat(claz, ["0", "1"], noise)

			strFeatures = list(map(lambda f: toStr(f, 2), features))
			rec =  genID(keyLen) + "," + ",".join(strFeatures) + "," + claz
			print (rec)

	def encodeDummy(self, fileName, extra):
		"""
		dummy var encoding
		"""
		catVars = {}
		catVars[1] = self.marStatus
		catVars[10] = self.loanTerm
		rSize = NFEAT_EXT if extra else NFEAT
		rSize += 2
		dummyVarGen = DummyVarGenerator(rSize, catVars, "1", "0", ",")
		for row in fileRecGen(fileName, None):
			newRow = dummyVarGen.processRow(row)
			print (newRow)

	def encodeLabel(self, fileName):
		"""
		label encoding
		"""
		catVars = {}
		catVars[1] = self.marStatus
		catVars[10] = self.loanTerm
		encoder = CatLabelGenerator(catVars, ",")
		for row in fileRecGen(fileName, None):
			newRow = encoder.processRow(row)
			print (newRow)

def mutatorOne(r):
	""" shifts and scales data  """
	inc = int(int(r[7]) * 0.9 - 10)
	r[7] = str(inc)
	
	debt = int(int(r[10]) * 1.6 + 20)
	r[10] = str(debt)
	return r
	
def mutatorTwo(r):
	cl = int(r[19])	
	idistr = GaussianRejectSampler(180,17) if cl == 1 else  GaussianRejectSampler(80,12)
	inc = int(idistr.sample())
	oinc = int(r[7])
	#print("current {}  new {}".format(oinc, inc))
	r[7] = str(inc)
	
	return r

def encodeRec(rec, encoder):
	""" encode record for categorical variables """
	typeSpec = "0:str,1:cat,2:int,3:int,4:int,5:int,6:float,7:float,8:int,9:int,10:cat,11:int,12:int,13:int,14:int"
	rec = getRecAsTypedRecord(rec, typeSpec)
	rec = encoder.processRow(rec)
	return rec
	
##########################################################################################
if __name__ == "__main__":
	op = sys.argv[1]

	if op == "generate" or op == "genOne" :
		"""  generate data """
		numLoans = int(sys.argv[2])
		loan = LoanApprove(numLoans)
		loan.generateOne()
	
	elif op == "genTwo":
		"""  generate data """
		numLoans = int(sys.argv[2])
		loan = LoanApprove(numLoans)
		noise = float(sys.argv[3])
		keyLen = int(sys.argv[4])
		addExtra = True if len(sys.argv) == 6 and sys.argv[5] == "extra" else False
		loan.generateTwo(noise, keyLen, addExtra)
	
	elif op == "encDummy":
		""" encode binary """
		fileName = sys.argv[2]
		extra = True if len(sys.argv) == 4 and sys.argv[3] == "extra" else False
		loan = LoanApprove()
		loan.encodeDummy(fileName, extra)
	
	elif op == "encLabel":
		""" encode label """
		fileName = sys.argv[2]
		loan = LoanApprove(numLoans)
		loan.encodeLabel(fileName)
	
	elif op == "nnTrain":
		""" tran neural network model """
		prFile = sys.argv[2]
		loModel = FeedForwardNetwork(prFile)
		loModel.buildModel()
		FeedForwardNetwork.batchTrain(loModel)
	
	elif op == "scaleParam":
		""" get scaling parameters and save them """
		prFile = sys.argv[2]
		dFile = sys.argv[3]
		loModel = FeedForwardNetwork(prFile)
		config = loModel.getConfig()
		scMethod = config.getStringConfig("common.scaling.method")[0]
		columns = config.getIntListConfig("train.data.fields")[0]
		columns = columns[:-1]
		spFile = config.getStringConfig("common.scaling.param.file")[0]
		if scMethod == "minmax":
			minMax = getFileColumnsMinMax(dFile, columns, "float")
			saveObject(minMax, spFile)
			print(minMax)
		else:
			exitWithMsg("invalid scaling method")	

	elif op == "calib":
		""" calibrate """
		prFile = sys.argv[2]
		clflier = FeedForwardNetwork(prFile)
		clflier.buildModel()
		ModelCalibration.findModelCalibration(clflier)
		
	elif op == "clsep":
		""" class separation performance metric using KS 2 sample"""
		prFile = sys.argv[2]
		clflier = FeedForwardNetwork(prFile)
		clflier.buildModel()
		
		FeedForwardNetwork.prepValidate(clflier)
		FeedForwardNetwork.validateModel(clflier)
		yPred = clflier.yPred.flatten()
		yPredNeg = list(map(lambda y : 1.0 - y, yPred))
		print(yPred[:4])
		print(yPredNeg[:4])
		print(type(yPred))

		expl = DataExplorer()
		expl.addListNumericData(yPred.tolist(),  "yp")
		expl.addListNumericData(yPredNeg,  "yn")
		expl.testTwoSampleKs("yp", "yn")
	
	elif op == "shsc":
		""" shift and scale data"""
		fpath = sys.argv[2]
		for rec in fileMutatedFieldsRecGen(fpath, mutatorTwo):
			print(",".join(rec))
			
	elif op == "presc":
		""" prescriptive analytic """
		optConfFile = sys.argv[1]
		mlConfFile = sys.argv[2]
		cfcCost = LoanCounterFacCost()
		optimizer = GeneticAlgorithmOptimizer(optConfFile, cfcCost)

	elif op == "lrobust":
		""" local robustness """
		prFile = sys.argv[2]
		clflier = FeedForwardNetwork(prFile)
		clflier.setVerbose(False)
		clflier.buildModel()
		
		fpath = sys.argv[3]
		nsamp = int(sys.argv[4])
		ncount = int(sys.argv[5])
		mr = ModelRobustness()
		mr.localPerformance(clflier, fpath, nsamp, ncount)
		
	elif op == "nnPred":
		""" tran neural network model """
		prFile = sys.argv[2]
		loModel = FeedForwardNetwork(prFile)
		loModel.buildModel()
		yp = FeedForwardNetwork.modelPredict(loModel)
		print(yp[:4])

	elif op == "predOne":
		""" tran neural network model """
		prFile = sys.argv[2]
		rec = sys.argv[3]
		typeSpec = "0:str,1:cat,2:int,3:int,4:int,5:int,6:float,7:float,8:int,9:int,10:cat,11:int,12:int,13:int,14:int"
		lt = len(typeSpec.split(","))
		lr = len(rec.split(","))
		#print (lr, lt)
		rec = getRecAsTypedRecord(rec, typeSpec, ",")
		
		catVars = {}
		marStatus = ["married", "single", "divorced"]
		loanTerm = ["7", "15", "30"]
		catVars[1] = marStatus
		catVars[10] = loanTerm
		dummyVarGen = DummyVarGenerator(lr, catVars, 1, 0)
		rec = dummyVarGen.processRow(rec)
		
		loModel = FeedForwardNetwork(prFile)
		loModel.buildModel()
		yp = FeedForwardNetwork.modelPredict(loModel, [rec])
		print(yp)

	elif op == "amtarg":
		""" make loan amount the target for approved cases turning into a regression data set """
		fileName = sys.argv[2]
		for rec in fileRecGen(fileName):
			if rec[19] == "1":
				amt = int(rec[11])
				income = int(rec[7])
				amt = int(0.5 * 2.5 * income + 0.5 * amt)
				nrec = list(rec[:11])
				nrec.extend(rec[12:19])
				nrec.append(str(amt))
				print(",".join(nrec))
	
	elif op == "confcal":
		""" calibration for conformal prediction """	
		prFile = sys.argv[2]
		confBound = float(sys.argv[3])
		calibFile = sys.argv[4]
		
		loModel = FeedForwardNetwork(prFile)
		loModel.buildModel()
		FeedForwardNetwork.prepValidate(loModel)
		res = FeedForwardNetwork.validateModel(loModel, True)
		ypair = res[0]
		print(ypair[:8])
		conPred = ConformalRegressionPrediction()
		conPred.calibrate(ypair, confBound)
		conPred.saveCalib(calibFile)
		
	elif op == "addnoiseic":
		""" add  noise to income and creditt score """
		filePath = sys.argv[2]
		ncount = 0
		for rec in fileRecGen(filePath):
			if isEventSampled(20):
				#rec[7] = str(int(addNoiseNum(int(rec[7]), UniformNumericSampler(-0.7, -0.4))))
				rec[7] = str(int(addNoiseNum(int(rec[7]), UniformNumericSampler(0.4, 0.8))))
				rec[14] = str(int(addNoiseNum(int(rec[14]), UniformNumericSampler(-0.4, -0.1))))
				ncount += 1
			print(",".join(rec))
		#print(ncount)
				
	elif op == "confpred":
		"""  conformal prediction  """
		prFile = sys.argv[2]
		calibFile = sys.argv[3]
		loModel = FeedForwardNetwork(prFile)
		loModel.buildModel()
		yp = FeedForwardNetwork.modelPredict(loModel)
		
		conPred = ConformalRegressionPrediction()
		conPred.restoreCalib(calibFile)
		for y in yp:
			res = conPred.getPredRange(y[0])
			print("{:.3f}\t{:.3f}\t{:.3f}\t{}\t{:.3f}".format(y[0], res["predRangeMin"], res["predRangeMax"], res["status"], res["confidence"]))

	elif op == "detood":
		""" detect ood data """
		prFile = sys.argv[2]
		daFile = sys.argv[3]
		lfilePath = sys.argv[4]
		avdFilePath = sys.argv[5]
		mode = sys.argv[6]
		assert mode =="gen" or mode == "det" , "invalid mode for ood detection"
		
		encoder = None
		loModel = FeedForwardNetwork(prFile)
		loModel.buildModel()
		FeedForwardNetwork.addForwardHook(loModel, 1)	
		
		if mode == "det":
			#build kd tree in detection mode
			fData = getFileAsFloatMatrix(lfilePath, list(range(8)))
			tree = KDTree(np.array(fData), leaf_size=4)
			
			ucb = float(sys.argv[7])
			avDists = getFileAsFloatColumn(avdFilePath)
			uci = int(ucb * len(avDists))
			ucv = avDists[uci]
			print("upper conf value " + formatFloat(6, ucv))
		
		yps = list()	
		for rec in fileRecGen(daFile):
			frec = rec[:-1]
			ya = int(rec[-1])
		
			if encoder is None:
				lr = len(frec)
				catVars = {}
				marStatus = ["married", "single", "divorced"]
				loanTerm = ["7", "15", "30"]
				catVars[1] = marStatus
				catVars[10] = loanTerm
				encoder = DummyVarGenerator(lr, catVars, 1, 0)
			frec = encodeRec(frec, encoder)
			yp = FeedForwardNetwork.modelPredict(loModel, [frec])
			print(frec[:5])
			print("predicted and actual {:.3f}  {:.3f}".format(float(yp[0][0]), ya))
			
			if mode == "gen":
				#latent data generation mode
				y = yp[0][0]
				yps.append(y)
			else:
				#ood pdetection mode
				lvalues = getLatValues()
				lv = lvalues[-1]
				dist, ind = tree.query(np.array([lv]), k=5)
				avDist = np.mean(dist[0])
				ood = avDist > ucv 
				print("av dist to neighbors {:.6f}   ood {}".format(avDist, ood))
		
		#save latent values	
		if mode == "gen":
			lvalues = getLatValues()
			with open(lfilePath, "w") as fh:
				for l, y in zip(lvalues, yps):
					r = toStrFromList(l, 6) + "," + formatFloat(3, y) + "\n"
					fh.write(r)
		
			#ave rage distance to neighboors in latent space
			lvalues = np.array(lvalues)
			tree = KDTree(lvalues, leaf_size=4)
			dist, ind = tree.query(lvalues, k=5)
			avDists = list()
			for d in dist:
				avDist = float(np.mean(d))
				avDists.append(avDist)
					
			avDists.sort()	
			writeFloatListToFile(avDists, 6, avdFilePath)
		
	
	else:
		exitWithMsg("unknow operation")



