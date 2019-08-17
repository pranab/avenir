#!/usr/bin/python

import os
import sys
from random import randint
import time
import uuid
import threading
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

def createAnomaly(high):
	if high:
		reading = randomFloat(120, 200)
	else:
		reading = randomFloat(60, 80)
	return reading
	
if __name__ == "__main__":
	op = sys.argv[1]
	
	#device stats
	if op == "stat":
		#normal mean 80 - 100 sd 1 - 5 
		#anomaly  mean 120 - 160 sd 1 - 5 
		numProds = int(sys.argv[2])
		mminTg = int(sys.argv[3])
		mmaxTg = int(sys.argv[4])
		sminTg = int(sys.argv[5])
		smaxTg = int(sys.argv[6])
		
		mminQu = int(sys.argv[7])
		mmaxQu = int(sys.argv[8])
		sminQu = int(sys.argv[9])
		smaxQu = int(sys.argv[10])
		for i in range(numProds):
			prId = genID(12)
			meanTg = randomFloat(mminTg, mmaxTg)
			sdTg = randomFloat(sminTg, smaxTg)
			meanQu = randomFloat(mminQu, mmaxQu)
			sdQu = randomFloat(sminQu, smaxQu)
			print "%s,%.3f,%.3f, %.3f, %.3f" %(prId, meanTg, sdTg, meanQu, sdQu)

	elif op == "gen":
		statFile = sys.argv[2]
		startNumDaysPast = int(sys.argv[3])
		endNumDaysPast = int(sys.argv[4]) if (len(sys.argv) == 5) else 0
		
		products = []
		for rec in fileRecGen(statFile, ","):
			ps = (rec[0], float(rec[1]), float(rec[2]), float(rec[3]), float(rec[4]))
			products.append(ps)
		
		numProds = len(products)
		distrsTg = list(map(lambda p: GaussianRejectSampler(p[1],p[2]), products))	
		distrsQu = list(map(lambda p: GaussianRejectSampler(p[3],p[4]), products))	

		curTime = int(time.time())
		startTime = curTime - (startNumDaysPast  + 1) * secInDay
		endTime = curTime - endNumDaysPast * secInDay
		morningBegSec = 6 * 60 * 60
		
		for i in range(numProds):
			pid = products[i][0]
			trTime = startTime
			while trTime < endTime:
				tg = distrsTg[i].sample()
				secIntoDay = trTime % secInDay
				if (secIntoDay < morningBegSec):
					tg *= randint(8, 16)
				trTime += tg
				qu = int(distrsQu[i].sample())
				if (qu < 1):
					qu = 1
				print "%s,%d,%d" %(pid, trTime, qu)
			