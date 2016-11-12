#!/usr/bin/python

import os
import sys
from random import randint
import time
import uuid
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

num_prod = int(sys.argv[1])
history_num_weeks = int(sys.argv[2])

prod_ids = []
fulfill_distr = []
for i in range(num_prod):
	prod_ids.append(genID(12))
	mean = randint(50, 80)
	stddev = randint(10, 20)
	fulfill_distr.append(GaussianRejectSampler(mean,stddev))
	
ms_per_day = 24 * 60 * 60 * 1000
ms_per_week = 7 * ms_per_day
now = curTimeMs()
past = now - (history_num_weeks + 5) * ms_per_week
past = (past / ms_per_week) * ms_per_week
cur_time = past

while (cur_time < now):
	for i in range(num_prod):
		prod_id = prod_ids[i] 
		if randint(0, 100) > 40:
			fulfill = 100
		else:
			fulfill = fulfill_distr[i].sample()
			if (fulfill < 20):
				 fulfill = 20
			if (fulfill > 100):
				fulfill = 100
		
		if (fulfill == 100):
			level = "F"
		elif (fulfill > 60):
			level = "P"
		else:
			level = "L"
					
		print "%s,%d,%s" %(prod_id, cur_time, level)
			
	cur_time += ms_per_week + randint(-10, 10)

