#!/usr/bin/python

import os
import sys
from random import randint
import time
from array import *
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

class SalesLead:
	def __init__(self, numLeads):
		self.numLeads = numLeads
		self.sourceDistr = CategoricalRejectSampler(("tradeShow", 80), ("webDownload", 60), ("referral", 100), ("advertisement", 40))
		self.contactTypeDistr = CategoricalRejectSampler(("canReccommend", 100), ("canDecide", 40))
		self.companySizeDistr = CategoricalRejectSampler(("small", 40), ("medium", 100), ("large", 60))
		self.numDaysDistr = GaussianRejectSampler(60,30)
		self.numMeetingsDistr = GaussianRejectSampler(5,2)
		self.numEmailsDistr = GaussianRejectSampler(10, 3)
		self.numWeSiteVisitsDistr = GaussianRejectSampler(5, 2)
		self.numDemosDistr = GaussianRejectSampler(3, 1)
		self.expRevDistr = GaussianRejectSampler(50000, 10000)
		self.proposalSentDistr = CategoricalRejectSampler(("Y", 40), ("N", 100))
		
		self.sourceScore = {"tradeShow" : 12 , "webDownload" : 10, "referral" : 20, "advertisement" : 6}
		self.contactTypScore = {"canReccommend" : 15 , "canDecide" : 25}
		self.companySizeScore = {"small" : 7 , "medium" : 12, "large" : 15}
		self.numDaysScore = StepFunction((1, 20, 2), (20, 50, 5), (50, 80, 7), (80, 120, 8))
		self.numMeetingsScore = StepFunction((0, 1, 1), (1, 5, 8), (5, 15, 9))
		self.numEmailsScore = StepFunction((0, 1, 1), (1, 7, 6), (7, 18, 8))
		self.numWebSiteVisitsScore = StepFunction((0, 1, 1), (1, 5, 5), (5, 12, 7))
		self.numDemosScore = StepFunction((0, 1, 1),(1, 3, 15), (3, 5, 20))
		self.expRevScore = StepFunction((1, 30000, 16), (30000, 60000, 13), (60000, 100000, 10))
		self.proposalSentScore = {"Y" : 18 , "N" : 7}
		
	def generate(self):
		convCount = 0
		for i in range(self.numLeads):
			id = genID(10)
			source = self.sourceDistr.sample()
			contactType = self.contactTypeDistr.sample()
			companySize = self.companySizeDistr.sample()
			numDays = int(self.numDaysDistr.sample())
			numDays = minLimit(numDays, 5)
			numMeetings = int(self.numMeetingsDistr.sample())
			numMeetings = minLimit(numMeetings, 0)
			numEmails = int(self.numEmailsDistr.sample())
			numEmails = minLimit(numEmails, 0)
			numWebSiteVisits = int(self.numWeSiteVisitsDistr.sample())
			numWebSiteVisits = minLimit(numWebSiteVisits, 0)
			numDemos = int(self.numDemosDistr.sample())
			numDemos = minLimit(numDemos, 0)
			expRev = self.expRevDistr.sample()
			expRev = minLimit(expRev, 30000)
			proposalSent = self.proposalSentDistr.sample()
			
			#conversion score
			score = 0
			score +=  self.sourceScore[source]
			score +=  self.contactTypScore[contactType]
			score +=  self.companySizeScore[companySize]
			score +=  self.numDaysScore.find(numDays)
			score +=  self.numMeetingsScore.find(numMeetings)
			score +=  self.numEmailsScore.find(numEmails)
			score +=  self.numWebSiteVisitsScore.find(numWebSiteVisits)
			score +=  self.numDemosScore.find(numDemos)
			score +=  self.expRevScore.find(expRev)
			score +=  self.proposalSentScore[proposalSent]
			if (score > 116 and randint(0, 100) > 5):
				converted = "1"
				convCount += 1
			else:
				converted = "0"
				
			print "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%s,%s" %(id,source, contactType, companySize, numDays, numMeetings, numEmails, numWebSiteVisits, numDemos, expRev, proposalSent, converted)
		#print "num converted %d" %(convCount)
		
	def generateDummyVars(self, file):
		catVars = {}
		catVars[1] = ("tradeShow", "webDownload", "referral", "advertisement")
		catVars[2] = ("canReccommend", "canDecide")
		catVars[3] = ("small", "medium", "large")
		catVars[10] = ("Y", "N")
		dummyVarGen = DummyVarGenerator(12, catVars, "1", "0", ",")
		fp = open(file, "r")
		for row in fp:
			newRow = dummyVarGen.processRow(row)
			print newRow.strip()
		fp.close()
			
			
##########################################################################################
op = sys.argv[1]
numLeads = int(sys.argv[2])
lead = SalesLead(numLeads)

if op == "generate":
	lead.generate()
elif op == "genDummyVar":
	file = sys.argv[3]
	lead.generateDummyVars(file)
			