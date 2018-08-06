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

if len(sys.argv) < 2:
	print "usage: ./cust_value.py <num_custs> "
	sys.exit()

num_custs = int(sys.argv[1])

genders = ["F", "F", "M", "M"]
visit_freqs = ["H", "M", "L", "M"]
zip_codes = []
all_zip_codes = []
values = ["T", "F"]

for i in range(4):
	num_zc = randint(10, 30)
	zc_list = []
	for j in range(num_zc):
		zc = genNumID(5) 	
		zc_list.append(zc)
		all_zip_codes.append(zc)
	zip_codes.append(zc_list)

num_zc = num_custs / 10
for i in range(num_zc):
	zc = genNumID(5) 	
	all_zip_codes.append(zc)

	
for i in range(num_custs):
	cust_id = genID(8)
	if randint(1, 100) < 70:
		gender = selectRandomFromList(genders)
		zc = selectRandomFromList(all_zip_codes)
		visit_feq = selectRandomFromList(visit_freqs)
		if randint(1, 100) < 80:
			value = "F"
		else:
			value = "T"
	else:
		c = randint(0, 3)
		gender = genders[c]
		zc = selectRandomFromList(zip_codes[c])
		visit_feq = visit_freqs[c]
		value = "T"
	print "%s,%s,%s,%s,%s" %(cust_id, gender, zc, visit_feq, value)
		
		