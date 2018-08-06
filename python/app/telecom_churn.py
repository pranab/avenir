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

if len(sys.argv) < 4:
	print "usage: ./telecom_churn.py <num_customers> <churn_rate> <error_rate> "
	sys.exit()

numCust = int(sys.argv[1])
churnRate = int(sys.argv[2])
errorRate = int(sys.argv[3])
threshold = 100 - errorRate
	
outPlanID = True

plans = ["plan A", "plan B"]
minUsageDistr = [GaussianRejectSampler(600,50), GaussianRejectSampler(1200,300)]
dataUsageDistr = [GaussianRejectSampler(200,50), GaussianRejectSampler(500,150)]
csCallDistr = [GaussianRejectSampler(4,1), GaussianRejectSampler(8,2)]
csEmailDistr = [GaussianRejectSampler(6,2), GaussianRejectSampler(10,3)]
networkSizeDistr = [GaussianRejectSampler(3,1), GaussianRejectSampler(6,2)]

for i in range(0, numCust):
	plan = selectRandomFromList(plans)
	probChurn = randint(1,100)
	if (probChurn < churnRate):
		#churning
		case = randint(1,4)
		
		churned = "Y"
		if (case == 1 or case == 4):
			#bad plan and too much usage
			plan = "plan A"
			minUsed = minUsageDistr[1].sample()
			dataUsed = dataUsageDistr[1].sample()
			csCall = csCallDistr[0].sample();
			csEmail = csEmailDistr[0].sample()
			network = networkSizeDistr[1].sample()
			planID = 1
		elif (case == 2):
			# too much CS calls
			plan = "plan B"
			minUsed = minUsageDistr[0].sample()
			dataUsed = dataUsageDistr[0].sample()
			csCall = csCallDistr[1].sample();
			if (csCall < 6):
				csCall = 6
			csEmail = csEmailDistr[1].sample()
			if (csEmail < 8):
				csEmail = 8
			network = networkSizeDistr[1].sample()
			planID = 2
		elif (case == 3):
			# too few people in network with the same subsciber
			plan = "plan B"
			minUsed = minUsageDistr[0].sample()
			dataUsed = dataUsageDistr[0].sample()
			minUsed = minUsed + 200
			dataUsed = dataUsed + 100
			csCall = csCallDistr[0].sample();
			csEmail = csEmailDistr[0].sample()
			network = networkSizeDistr[0].sample()
			planID = 2
			
		if (randint(1,100) < threshold):
			churn = 1
		else:
			churn = 0
	else:
		#non churning			
		churned = "N"
		if (plan == "plan A"):
			planID = 1
		else:
			planID = 2
		minUsed = minUsageDistr[planID - 1].sample()
		dataUsed = dataUsageDistr[planID -1].sample()
		csCall = csCallDistr[0].sample();
		if (csCall > 2):
			csCall = 2
		csEmail = csEmailDistr[0].sample()
		if (csEmail > 3):
			csEmail = 3
		network = networkSizeDistr[1].sample()
		if (randint(1,100) < threshold):
			churn = 0
		else:
			churn = 1
	if (outPlanID):		
		print "%d,%d,%d,%d,%d,%d,%d"  %(planID,minUsed,dataUsed,csCall,csEmail,network,churn)
	else:
		print "%s,%d,%d,%d,%d,%d,%d"  %(plan,minUsed,dataUsed,csCall,csEmail,network,churn)
	
	
