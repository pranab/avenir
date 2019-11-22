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
from array import *
sys.path.append(os.path.abspath("../lib"))
from mlutil import *
from util import *
from sampler import *

class LoanApprove:
	def __init__(self, numLoans):
		self.numLoans = numLoans
		self.marStatus = ["married", "single", "divorced"]
		self.loanTerm = ["7", "15", "30"]
		
	def initOne(self):
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

	# ad hoc
	def generateOne(self):
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
		
		
	#ancestral sampling
	def generateTwo(self, noise, keyLen):
		self.initTwo()
		
		#error
		erDistr = GaussianRejectSampler(0, noise)
	
		#sampler
		sampler = AncestralSampler(self.approvDistr, self.featCondDister, 11)

		for i in range(self.numLoans):
			(claz, features) = sampler.sample()
		
			# add noise
			features[0] = addNoiseCat(features[0], self.marStatus, noise)
			features[1] = addNoiseCat(features[1], ["1", "2", "3"], noise)
			features[2] = addNoiseCat(features[2], ["1", "2", "3"], noise)
			features[3] = addNoiseCat(features[3], ["0", "1"], noise)
			features[4] = int(addNoiseNum(features[4], erDistr))
			features[5] = float(addNoiseNum(features[5], erDistr))
			features[6] = float(addNoiseNum(features[6], erDistr))
			features[7] = int(addNoiseNum(features[7], erDistr))
			features[8] = int(addNoiseNum(features[8], erDistr))
			features[9] = addNoiseCat(features[9], self.loanTerm, noise)
			features[10] = int(addNoiseNum(features[10], erDistr))

			claz = addNoiseCat(claz, ["0", "1"], noise)

			strFeatures = list(map(lambda f: toStr(f, 2), features))
			rec =  genID(keyLen) + "," + ",".join(strFeatures) + "," + claz
			print (rec)

	# dummy var encoding
	def encodeDummy(self, fileName):
		catVars = {}
		catVars[1] = self.marStatus
		catVars[10] = self.loanTerm
		rSize = 13
		dummyVarGen = DummyVarGenerator(rSize, catVars, "1", "0", ",")
		for row in fileRecGen(fileName):
			newRow = dummyVarGen.processRow(row)
			print (newRow)

	# label encoding
	def encodeLabel(self, fileName):
		catVars = {}
		catVars[1] = self.marStatus
		catVars[10] = self.loanTerm
		encoder = CatLabelGenerator(catVars, ",")
		for row in fileRecGen(fileName):
			newRow = encoder.processRow(row)
			print (newRow)

##########################################################################################
if __name__ == "__main__":
	op = sys.argv[1]
	numLoans = int(sys.argv[2])
	loan = LoanApprove(numLoans)

	if op == "generate" or op == "genOne" :
		loan.generateOne()
	elif op == "genTwo":
		noise = float(sys.argv[3])
		keyLen = int(sys.argv[4])
		loan.generateTwo(noise, keyLen)
	elif op == "encDummy":
		fileName = sys.argv[3]
		loan.encodeDummy(fileName)
	elif op == "encLabel":
		fileName = sys.argv[3]
		loan.encodeLabel(fileName)
	else:
		print ("unknow operation")



