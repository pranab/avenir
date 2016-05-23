#!/Users/pranab/Tools/anaconda/bin/python

import os
import sys
import sys
import random 
import time
import math
import numpy as np
sys.path.append(os.path.abspath("../lib"))
from sampler import *


num_samples = int(sys.argv[1])
inv = int (sys.argv[2])
profit_per_unit = float(sys.argv[3])
holding_cost_per_unit = float(sys.argv[4])
back_ord_cost_per_unit = float(sys.argv[5])

demand_distr = MetropolitanSampler(25, 0, 10, 7,12,22,16,13,10,8,12,19,23,27,34,25,18,12,5,2)
back_ord_distr = GaussianRejectSampler(0.35, 0.10)
earnings = np.zeros(num_samples)
burn_in_sample = 0.1 * num_samples

inv_step = 5
num_inv = 12

# num of inventory levels
for i in range(num_inv):
	excess_cnt = 0
	deficit_cnt = 0
	
	#num of simulation sample
	demand_distr.transCount = 0
	for j in range(num_samples):
		dem = demand_distr.sample()
		#print "demand %d" %(dem)
		if inv >= dem:
			#excess inventory
			#print "excess inventory"
			earning = dem * profit_per_unit
			hld_cost = (inv - dem) * holding_cost_per_unit
			earning -= hld_cost
			excess_cnt += 1
		else:
			#not enough inventory
			#print "deficit inventory"
			earning = inv * profit_per_unit
			unfulfilled = dem - inv
			back_ordered = unfulfilled * back_ord_distr.sample()
			unfulfilled -= back_ordered
			net_cost = unfulfilled * profit_per_unit + back_ordered * back_ord_cost_per_unit
			earning += back_ordered * profit_per_unit
			earning -= net_cost
			deficit_cnt += 1
		earnings[j] = earning
	
	mean_earning = earnings[burn_in_sample:].mean()
	out = (inv, mean_earning, excess_cnt, deficit_cnt, demand_distr.transCount)
	print "inventory %d average earning %.3f excess count %d deficit count %d transition count %d" %out
	
	inv += inv_step
		
		

