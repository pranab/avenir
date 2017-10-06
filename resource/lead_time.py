#!/usr/bin/python

import os
import sys
from random import randint
import time
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

num_recs = int(sys.argv[1])

products = []
high_lead_time_products = set()
num_prods = 50
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
high_lead_time_months = {"08","10", "11"}

for i in range(num_prods):
	prod = genID(10)
	products.append(prod)
	if randint(0, 100) < 30:
		high_lead_time_products.add(prod)
		#print "high lead time %s" %(prod)
	
order_id = genID(12)
month = selectRandomFromList(months)
this_month = {month}
j = 0
num_item = randint(5, 15)
count = 0
for i in range(num_recs):
	prod = selectRandomFromList(products)
	quant = randint(100, 1000)
	score = randint(5, 10)
	this_prod = {prod}
	if high_lead_time_products.issuperset(this_prod):
		score += 40
	
	if high_lead_time_months.issuperset(this_month):
		score += 30
	
	if quant > 800:
		score += 30
	elif quant > 500:
		score += 20
		
	if score > 60:
		status = "T"
		count += 1
	else:
		status = "F"
		
	print "%s,%s,%d,%s,%s" %(order_id,prod,quant,month,status)	
	
	j += 1
	if j % num_item == 0:
		order_id = genID(12)
		month = selectRandomFromList(months)
		this_month = {month}
		j = 0
		num_item = randint(5, 15)
		
#print  "true count %d" %(count)	
