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


if __name__ == "__main__":
	numApps = int(sys.argv[1])
	numModels = int(sys.argv[2])
	numAdvt = int(sys.argv[3])
	numZC = int(sys.argv[4])
	numDays = int(sys.argv[5])

	apps = genIdList(numApps, 8)
	adverts = genIdList(numAdvt, 10)
	zipCodes = genNumIdList(numZC, 5)
	oss = ["Android", "iOS"]
	models = dict()
	models["Android"] = genIdList(int(numModels * .6), 8)
	models["iOS"] = genIdList(int(numModels * .4), 8)
	
	curTime = int(time.time())
	pastTime = curTime - (numDays + 1) * secInDay
	sampTime = pastTime
	
	while sampTime < curTime:
		impressionID = genID(16)
		os = selectRandomFromList(oss)
		model = selectRandomFromList(models[os])
		app = selectRandomFromList(apps)
		advt = selectRandomFromList(adverts)
		zc = selectRandomFromList(zipCodes)
		tapped = 1 if isEventSampled(20) else 0
		print "%s,%s,%s,%s,%s,%s,%d" %(impressionID, os, model, app, advt, zc, tapped)
		sampTime += randint(5, 30)		
