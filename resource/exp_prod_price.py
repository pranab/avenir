#!/usr/bin/python

import os
import sys
from random import randint
import time
import uuid
from array import *
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

class DiscountOffer:
	def __init__(self, leadTime, discountRate, invetoryMean, invetoryStdDev, demandMean, demandStdDev):
		self.leadTime = leadTime
		self.discountRate = discountRate
		self.invDistr = GaussianRejectSampler(invetoryMean,invetoryStdDev)
		self.demanDistr = GaussianRejectSampler(demandMean,demandStdDev)
	
	@classmethod
	def initCls(cls, numProds, leadTimes, discounts):
		cls.numProds = numProds
		cls.leadTimes = leadTimes
		cls.discounts = discounts
		cls.maxLeadTime = leadTimes[len(leadTimes) - 1]
	
	@classmethod
	def createDiscounts(cls):
		for i in range(cls.numProds):
			pid = genID(8)
			for j in cls.leadTimes:
				for k in cls.discounts:
					item = str(j) + ":" + str(k)
					print "%s,%s"  %(pid,item)
	
	@classmethod
	def createDistrModel(cls, prodFilePath):
		fp = open(prodFilePath, "r")
		lastPid = ""
		invStat = {}
		demStat = {}

		for line in fp:
			items = line.split(",")
			pid = items[0]
			ld = items[1]
			discnt = ld.split(":")
			leadTime = int(discnt[0])
			discount = int(discnt[1])
			
			if (not pid == lastPid):
				cost = randomFloat(10.0, 50.0)
				price = cost * randomFloat(1.03, 1.08)
				maxInv = randomFloat(500.0, 1000.0)
				for l in cls.leadTimes:
					iMean = maxInv  * randomFloat(0.95, 1.05) * l / cls.maxLeadTime
					iStdDev = randomFloat(50.0, 100.0)
					invStat[l] = (iMean, iStdDev)
					for d in cls.discounts:
						dMean = iMean * randomFloat(0.9, 1.1)
						dStdDev = randomFloat(20.0, 150.0)
						dStat = demStat.get(l, {})	
						demStat[l] = dStat
						dStat[d] = (dMean, dStdDev)
				lastPid = pid
			
			iStat = invStat[leadTime]	
			dStat = demStat[leadTime][discount]
			print "%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" %(pid, ld.strip(), cost, price, iStat[0], iStat[1], dStat[0], dStat[1])				
			
		fp.close()
						
	def sampleInv():
		return self.invDistr.sample()

	def sampleDemand():
		return self.demandDistr.sample()

class PerishableProduct:
	def __init__(self):
		self.cost = float(randint(10, 50))
		self.price = self.cost * (1.03 + 0.1 * random())
		self.offers = []
	
	def addDiscountOffer(self, offer):
		self.offers.append(offer)	


	
		
##################################
op = sys.argv[1]
numProds = int(sys.argv[2])
leadTimes = array('i', [3,5])
discounts = array('i', [5,10])

DiscountOffer.initCls(numProds, leadTimes, discounts)
if (op == "discounts"):
	DiscountOffer.createDiscounts()		
elif (op == "sampModel"):
	prodFilePath = sys.argv[3]
	DiscountOffer.createDistrModel(prodFilePath)
	