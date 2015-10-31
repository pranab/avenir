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

numCust = int(sys.argv[1])

plans = ["plan A", "plan B"]
minUsageDistr = [GaussianRejectSampler(600,50), GaussianRejectSampler(1200,300)]
dataUsageDistr = [GaussianRejectSampler(200,50), GaussianRejectSampler(500,150)]
csCallDistr = [GaussianRejectSampler(4,1), GaussianRejectSampler(8,2)]
csEmailDistr = [GaussianRejectSampler(6,2), GaussianRejectSampler(10,3)]
networkSizeDistr = [GaussianRejectSampler(3,1), GaussianRejectSampler(6,2)]

for i in range(0, numCust):
	plan = selectRandomFromList(plans)
	probChurn = randint(1,99)
	if (probChurn > 80):
		#churning
		case = randin(1.3)
		churned = "Y"
		if (case == 1):
			#bad plan and too much usage
			plan = "plan A"
			minUsed = minUsageDistr[1].sample()
			dataUsed = dataUsageDistr[1].sample()
			csCall = csCallDistr[0].sample();
			csEmail = csEmailDistr[0].sample()
			network = networkSizeDistr[0].sample()
		elif (case == 2):
			# too much CS calls
			plan = "plan B"
			minUsed = minUsageDistr[1].sample()
			dataUsed = dataUsageDistr[1].sample()
			csCall = csCallDistr[1].sample();
			if (csCall < 6):
				csCall = 6
			csEmail = csEmailDistr[1].sample()
			if (csEmail < 8):
				csEmail < 8
			network = networkSizeDistr[0].sample()
		elif (case == 3):
			# too few people in network with the same subsciber
			plan = "plan B"
			minUsed = minUsageDistr[1].sample()
			dataUsed = dataUsageDistr[1].sample()
			minUsed = minUsed + 200
			dataUsed = dataUsed + 100
			csCall = csCallDistr[0].sample();
			csEmail = csEmailDistr[0].sample()
			network = networkSizeDistr[0].sample()
	else:
		#non churning			
		churned = "N"
		if (plan == "plan A"):
			p = 0
		else:
			p = 1
		minUsed = minUsageDistr[p].sample()
		dataUsed = dataUsageDistr[p].sample()
		csCall = csCallDistr[0].sample();
		if (csCall > 2):
			csCall = 2
		csEmail = csEmailDistr[0].sample()
		if (csEmail > 3):
			csEmail = 3
		network = networkSizeDistr[1].sample()
			
	print "%s,%d,%d,%d,%d,%d"  %(plan,minUsed,dataUsed,csCall,csEmail,network)
