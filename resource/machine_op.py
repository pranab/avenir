#!/usr/bin/python

import os
import sys
from random import randint
import time
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

num_recs = int(sys.argv[1])

age_distr = GaussianRejectSampler(60,15)
time_sice_last_maint_distr = GaussianRejectSampler(6,2)
first_freq_distr = GaussianRejectSampler(6000,200)
first_freq_amp_distr = GaussianRejectSampler(1.2, 0.2)
second_freq_distr = GaussianRejectSampler(8000,100)
second_freq_amp_distr = GaussianRejectSampler(1.0,0.1)

first_freq_norm_distr = GaussianRejectSampler(3000,200)
second_freq_norm_distr = GaussianRejectSampler(4400,100)

count = 0
for i in range(num_recs):
	rid = genID(12)
	age = age_distr.sample()
	time_sice_last_maint = time_sice_last_maint_distr.sample()
	num_breakdowns = 0
	if randint(0, 100) > 80:
		num_breakdowns = randint(0, 2)
	if randint(0, 100) > 90:
		first_freq = first_freq_distr.sample()
		second_freq = second_freq_distr.sample()
	else:
		first_freq = first_freq_norm_distr.sample()
		second_freq = second_freq_norm_distr.sample()
		
	first_freq_amp = first_freq_amp_distr.sample()
	second_freq_amp = first_freq_amp_distr.sample()
	
	pr = 0
	if age > 90:
		pr = pr + 10
	elif age > 80:
		pr = pr + 6
		
	if time_sice_last_maint > 10:
		pr = pr + 8
	elif time_sice_last_maint > 8:
		pr = pr + 6
	
	if num_breakdowns > 0:
		pr = pr + 20
	
	if first_freq > 6200:
		pr = pr + 26
	elif first_freq > 5800:
		pr = pr + 18
	
	if first_freq_amp > 1.4:
		pr = pr + 12
		
	if second_freq > 8200:
		pr = pr + 20
	elif second_freq > 7800:
		pr = pr + 16
	
	if second_freq_amp > 1.1:
		pr = pr + 8

	if pr > randint(40, 50):
		status = 1
		count = count + 1
	else:
		status = -1
		
	print "%s,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%d" %(rid,age,time_sice_last_maint,num_breakdowns,first_freq,first_freq_amp,second_freq,second_freq_amp,status)	
	
#print "positive class count %d" %(count)

