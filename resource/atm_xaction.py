#!/usr/bin/python

import os
import sys
from random import randint
import time
import uuid
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

num_atm = int(sys.argv[1])
history_num_days = int(sys.argv[2])
xaction_bucket = int(sys.argv[3])

atm_ids = []
for i in range(num_atm):
	atm_ids.append(genID(12))

xaction_distr = [GaussianRejectSampler(40,10), GaussianRejectSampler(60,15)]
ms_per_day = 24 * 60 * 60 * 1000
ms_per_week = 7 * ms_per_day
now = curTimeMs()
past = now - history_num_days * ms_per_day
past = (past / ms_per_day) * ms_per_day
cur_time = past

while (cur_time < now):
	for i in range(num_atm):
		atm_id = atm_ids[i]
		day = cur_time % ms_per_week
		day /= ms_per_day
		if day < 5:
			xactions = xaction_distr[0].sample()
		else:
			xactions = xaction_distr[1].sample()
		trans = int(xactions + xaction_bucket) / xaction_bucket
		trans *= xaction_bucket
		print "%s,%d,%d" %(atm_id, cur_time, trans)
			
	cur_time += ms_per_day
		
		

