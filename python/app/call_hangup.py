#!/usr/bin/python

import os
import sys
from random import randint
import time
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

num_calls = int(sys.argv[1])

cust_types = ["business", "residence"]
res_issue_type = ["internet", "cable", "billing", "other"]
biz_issue_type = ["internet", "billing", "other"]
time_of_day =["AM", "PM"]
hold_time_distr = {"AM" : GaussianRejectSampler(500,80), "PM" : GaussianRejectSampler(400,60)}
area_codes = [ 408,607,336,267,646,760,615,980,828,385,941,305,971,510,574,620,507,540,206,262,847,  \
941,470,323,630,615,346,216, 920,903,423,614,440,419,832,678,608,678,571,248,321,301,630,719,209, \
770,615,971,937,703]

def hangup(hold_time, threshold):
	if (hold_time > threshold):
		if (randint(0,100) > 20):
			hungup = "T"
		else:
			hungup = "F"
	else:
		if (randint(0,100) > 10):
			hungup = "F"
		else:
			hungup = "T"
	return hungup


for i in range(num_calls):
	cust_id = genID(10)
	cust_type = selectRandomFromList(cust_types)
	if (cust_type == "residence"):
		issue = selectRandomFromList(res_issue_type)
	else:
		issue = selectRandomFromList(biz_issue_type)
	area_code = selectRandomFromList(area_codes)
	tod = selectRandomFromList(time_of_day)
	hold_time = int(hold_time_distr[tod].sample())
	
	threshold = 180
	if (cust_type == "business"):
		if (issue == "internet"):
			threshold = 450
		if (issue == "billing"):
			threshold = 300
	else:
		if (issue == "internet"):
			threshold = 350
		if (issue == "billing"):
			threshold = 250
	hungup = hangup(hold_time, threshold)
	print "%s,%s,%d,%s,%s,%d,%s" %(cust_id, cust_type, area_code, issue, tod, hold_time, hungup)	
    
