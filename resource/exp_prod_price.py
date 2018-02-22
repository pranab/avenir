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

class PerishableProduct:
	def __init__(self, pid, cost, price):
		self.pid = pid
		self.cost = cost
		self.price = price
		self.profit = price - cost
		self.invDistr = {}
		self.demDistr = {}
	
	def setInvStat(self,leadTime,invMean,invStdDev):
		iDistr = self.invDistr.get(leadTime, GaussianRejectSampler(invMean,invStdDev))
		self.invDistr[leadTime] = iDistr
		
	def setDemStat(self, leadTime, discount, demMean, demStdDev):
		dDistrMap = self.demDistr.get(leadTime, {})	
		self.demDistr[leadTime] = dDistrMap
		dDistr = dDistrMap.get(discount, GaussianRejectSampler(demMean,demStdDev))
		dDistrMap[discount] = dDistr
	
	def sampleInv(self, leadTime):
		iDistr = self.invDistr[leadTime]
		return int(iDistr.sample())
	
	def sampleDem(self, leadTime, discount):
		dDistr = self.demDistr[leadTime][discount]
		return int(dDistr.sample())


class DiscountOffer:
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
				
	@staticmethod
	def sampleReward(modelFilePath, decFilePath):
		#load sampling model for each product
		fp = open(modelFilePath, "r")
		lastPid = ""
		prodMap = {}
		
		for line in fp:
			items = line.split(",")
			pid = items[0]
			ld = items[1]
			discnt = ld.split(":")
			leadTime = int(discnt[0])
			discount = int(discnt[1])
			cost = float(items[2])
			price = float(items[3])
			invMean = float(items[4])
			invStdDev = float(items[5])
			demMean = float(items[6])
			demStdDev = float(items[7])
			if (not pid == lastPid):
				prod = PerishableProduct(pid, cost, price)
				prodMap[pid] = prod
				lastPid = pid
			
			prod.setInvStat(leadTime,invMean,invStdDev) 
			prod.setDemStat(leadTime, discount, demMean, demStdDev)
		fp.close()

		#profit or loss for each product lead time discount decision
		fp = open(decFilePath, "r")
		lastPid = ""
		
		for line in fp:
			items = line.split(",")
			pid = items[0]
			ld = items[1].strip()
			discnt = ld.split(":")
			leadTime = int(discnt[0])
			discount = int(discnt[1])
			
			prod = prodMap[pid]
			inv = prod.sampleInv(leadTime)	
			dem = prod.sampleDem(leadTime, discount)
			if (dem > inv):
				#everything sold
				profit = inv  * prod.profit
			else:
				#partial inventory sold
				profit = dem * prod.profit
				loss = (inv - dem) * prod.cost
				profit = profit - loss
			profit = profit + 8000
			print "%s,%s,%.2f" %(pid, ld, profit)
		fp.close()
			
		
##########################################################################################
op = sys.argv[1]
numProds = int(sys.argv[2])
leadTimes = array('i', [3,5])
discounts = array('i', [5,10])

DiscountOffer.initCls(numProds, leadTimes, discounts)
if (op == "createDiscounts"):
	DiscountOffer.createDiscounts()		
elif (op == "createSampModel"):
	prodFilePath = sys.argv[3]
	DiscountOffer.createDistrModel(prodFilePath)
elif (op == "sampleReward"):
	modelFilePath = sys.argv[3]
	decFilePath = sys.argv[4]
	DiscountOffer.sampleReward(modelFilePath, decFilePath)
	