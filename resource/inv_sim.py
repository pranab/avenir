#!/Users/pranab/Tools/anaconda/bin/python

import os
import sys
import sys
import random 
import time
import math
import numpy as np
import math
import jprops
sys.path.append(os.path.abspath("../lib"))
from sampler import *
from support import *

def eval_earning_inventory():
	print "earning for different inventory"
	earnings = np.zeros(sample_size)

	# num of inventory levels
	for i in range(num_inv):
		print "*** next inventory %d ***" %(inv)
		excess_cnt = 0
		deficit_cnt = 0
	
		#num of simulation samples
		demand_distr.initialize()
		for s in range(sample_size):
			dem = demand_distr.sample()
			(earning, in_excess) = get_earning(dem)
			earnings[s] = earning
			if in_excess:
				excess_cnt += 1
			else:
				deficit_cnt += 1
	
			earnings_stable = earnings[burn_in_sample:]
			if earning_stat == "mean":
				mean_earning = earnings_stable.mean()
	
		if verbose:
			err = earnings_stable.std() / sqr_sample
			out = (inv, mean_earning, err, excess_cnt, deficit_cnt, demand_distr.transCount)
			print "inventory %d average earning %.2f error %.3f excess count %d deficit count %d transition count %d" %out
		else:
			print "inventory %d average earning %.2f error %.3f excess count %d deficit count %d transition count %d" %(inv, mean_earning)
			
	
		inv += inv_step
	
# evaluate sample size effect
def eval_sample_size():
	print "sample size analysis" 
	for samp_size in list_sample_size:
		earnings = np.zeros(samp_size)
		demand_distr.initialize()
		excess_cnt = 0
		deficit_cnt = 0

		for s in range(samp_size):
			dem = demand_distr.sample()
			(earning, in_excess) = get_earning(dem)
			earnings[s] = earning
			if in_excess:
				excess_cnt += 1
			else:
				deficit_cnt += 1
			
		earnings_stable = earnings[burn_in_sample_size:]
		mean_earning = earnings_stable.mean()
		std_dev_earning = earnings_stable.std()
		#print "samp size %d burn in samp size %d" %(samp_size, burn_in_sample_size)
		error = std_dev_earning / math.sqrt(samp_size - burn_in_sample_size)
		print "sample size %d earning mean %.2f  earning mean std dev %.3f" %(samp_size, mean_earning, error)
		
		
# evaluate burn in sample size effect
def eval_burn_in_size():
	print "burn sample size analysis" 
	samp_size = sample_size
	prev_burn_in_sample_size = -1
	for burn_in_sample_size in list_burn_in_sample_size:
		if prev_burn_in_sample_size > 0:
			samp_size += (burn_in_sample_size - prev_burn_in_sample_size)
		earnings = np.zeros(samp_size)
		demand_distr.initialize()
		excess_cnt = 0
		deficit_cnt = 0
		
		for s in range(samp_size):
			dem = demand_distr.sample()
			(earning, in_excess) = get_earning(dem)
			earnings[s] = earning
			if in_excess:
				excess_cnt += 1
			else:
				deficit_cnt += 1
		
		earnings_stable = earnings[burn_in_sample_size:]
		mean_earning = earnings_stable.mean()
		std_dev_earning = earnings_stable.std()
		error = std_dev_earning / math.sqrt(samp_size - burn_in_sample_size)
		print "sample size %d earning mean %.3f  earning mean std dev %.3f" %(samp_size, mean_earning, error)
		prev_burn_in_sample_size = burn_in_sample_size
		
def get_earning(dem):
	in_excess = False
	if inv >= dem:
		#excess inventory
		earning = dem * profit_per_unit
		hld_cost = (inv - dem) * holding_cost_per_unit
		earning -= hld_cost
		in_excess = True
	else:
		#not enough inventory
		earning = inv * profit_per_unit
		unfulfilled = dem - inv
		back_ordered = unfulfilled * back_ord_distr.sample()
		unfulfilled -= back_ordered
		net_cost = unfulfilled * profit_per_unit + back_ordered * back_ord_cost_per_unit
		earning += back_ordered * profit_per_unit
		earning -= net_cost
	return (earning, in_excess)

########## main #################
configs = get_configs(sys.argv[1])
op = sys.argv[2]

inv = int(configs["inv.size"])
profit_per_unit = float(configs["profit.per.unit"])
holding_cost_per_unit = float(configs["holding.cost.per.unit"])
proposal_distr_std = int(configs["proposal.distr.std"])
demand_distr_start = int(configs["demand.distr.start"])
demand_distr_bin_width = int(configs["demand.distr.bin.width"])
demand_distr = get_int_array(configs["demand.distr"])
#7,12,22,16,13,10,8,12,19,23,27,34,25,18,12,5,2
demand_distr = MetropolitanSampler(proposal_distr_std, demand_distr_start, demand_distr_bin_width, demand_distr)
back_order_fraction_distr_mean = float(configs["back.order.distr.mean"])
back_order_fraction_distr_std = float(configs["back.order.distr.std"])
back_ord_distr = GaussianRejectSampler(back_order_fraction_distr_mean, back_order_fraction_distr_std)
back_ord_cost_per_unit = float(configs["back.order.cost.per.unit"])
verbose = configs["output.verbose"] == "true"


if  op == "samp_size":
	if "," in configs["sample.size"]: 
		list_sample_size = get_int_array(configs["sample.size"])
	else:
		sample_size = int(configs["sample.size"])
		sample_size_step = int(configs["sample.size.step"])
		num_sample_size = int(configs["num.sample.size"])
		list_sample_size = build_array(sample_size, sample_size_step, num_sample_size)
		
	burn_in_sample_size = int(configs["burn.in.sample.size"])
	eval_sample_size()
elif op == "burinin_size":
	sample_size = int(configs["sample.size"])
	
	if "," in configs["burn.in.sample.size"]: 
		list_burn_in_sample_size = get_int_array(configs["burn.in.sample.size"])
	else:
		burn_in_sample_size = int(configs["burn.in.sample.size"])
		burn_in_sample_size_step = int(configs["burn.in.sample.size.step"])
		burn_in_num_sample_size = int(configs["burn.in.num.sample.size"])
		list_burn_in_sample_size = build_array(burn_in_sample_size, burn_in_sample_size_step, burn_in_num_sample_size)
else:
	sample_size = int(configs["sample.size"])
	burn_in_sample_size = int(configs["burn.in.sample.size"])
	inv_step = int(configs["inv.step"])
	num_inv = int(configs["num.inv"])
	earning_stat = configs["earning.stat"]
	sqr_sample = math.sqrt(sample_size - burn_in_sample_size)
	eval_earning_inventory()
