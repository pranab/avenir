#!/usr/bin/python

import os
import sys
from random import randint
import time
from array import *
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

class LoanApprove:
	def __init__(self, numLoans):
		self.numLoans = numLoans
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

	def generate(self):
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

			print "%s,%s,%d,%d,%s,%d,%d,%d,%d,%d,%d,%s,%d" %(id, married, numChild, edu, selfEmployed, income,\
			numYearsExp, outstandingLoan, loanAm, loanTerm, credScore, zipCode, approved)

		#print "positive count " + str(posCount)


##########################################################################################
op = sys.argv[1]
numLoans = int(sys.argv[2])
loan = LoanApprove(numLoans)

if op == "generate":
	loan.generate()


