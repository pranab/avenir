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

if len(sys.argv) < 3:
	print "usage: ./cust_seg.py <num_customers>  <noise_level> "
	sys.exit()

num_cust = int(sys.argv[1])
noise_level = int(sys.argv[2])
threshold = []
clust_population = 100 - noise_level
threshold.append((clust_population * 40) / 100)
threshold.append((clust_population * 70) / 100)
threshold.append(clust_population)
	

num_visits_distr = [GaussianRejectSampler(15,3), GaussianRejectSampler(8,2), GaussianRejectSampler(20,5)]
visit_dur_distr = [GaussianRejectSampler(10,2), GaussianRejectSampler(20,3), GaussianRejectSampler(10,3)]
cust_id_numeric = True
cust_id = 1000000


for i in range(0, num_cust):
	case = randint(1,100)
	if cust_id_numeric:
		cust_id += 1
	else:
		cust_id = genID(8)
	if (case < threshold[0]):
		num_visit = num_visits_distr[0].sample()
		visit_dur = visit_dur_distr[0].sample()
		time_of_visit = 2
		num_xaction = int(num_visit * (0.4 + random.random() * 0.2))
		amount = num_xaction * 80 * (0.4 + random.random() * 0.3)
	elif (case < threshold[1]):
		num_visit = num_visits_distr[1].sample()
		visit_dur = visit_dur_distr[1].sample()
		time_of_visit = 3
		num_xaction = int(num_visit * (0.3 + random.random() * 0.3))
		amount = num_xaction * 100 * (0.9 + random.random() * 0.5)
	elif (case < threshold[2]):
		num_visit = num_visits_distr[2].sample()
		visit_dur = visit_dur_distr[2].sample()
		time_of_visit = 3
		num_xaction = int(num_visit * (0.5 + random.random() * 0.2))
		amount = num_xaction * 50 * (0.5 + random.random() * 0.5)
	else:
		num_visits = randint(1,30)
		visit_dur = randint(2,40)
		time_of_visit = randint(0,3)
		num_xaction = int(num_visit * (0.3 + random.random() * 0.5))
		amount = num_xaction * 50 * (0.2 + random.random() * 0.6)
	
	if cust_id_numeric:
		print "%d,%d,%d,%d,%d,%.3f"  %(cust_id,num_visit,visit_dur,time_of_visit,num_xaction,amount)
	else:
		print "%s,%d,%d,%d,%d,%.3f"  %(cust_id,num_visit,visit_dur,time_of_visit,num_xaction,amount)
	
	