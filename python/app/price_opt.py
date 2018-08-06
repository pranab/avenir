#!/usr/bin/python

import sys
from random import randrange

# create price data and average price revenue 
def create_price(prod_count, stat_file):
	fp = open(stat_file, "w")
	for pd in range(1, prod_count):
		prod_id = randrange(1000000, 8000000)
		num_price = randrange(6,12)
		price_delta = randrange(2,4)
		price = randrange(10,80)
		rev = randrange(10000, 30000)
		rev_delta = randrange(500, 1500)
		half_way =  num_price / 2 + randrange(-2,2)
		for pr in range(1, num_price):
			print "%d,%d,0,0,0" %(prod_id, price)
			l = "%d,%d,%d\n" %(prod_id, price, rev)
			fp.write(l)
			price += price_delta
			if (pr < half_way):
				rev += rev_delta + randrange(-20,20)
			else:
				rev -= rev_delta + randrange(-20,20)
	fp.close()

# generate initial revenue for all items		
def create_init_return(price_file, quant_ord):
	fp = open(price_file, "r")
	for line in fp:
		items = line.split(",")
		print "%s,%s,%d,0,0,0,0,0" %(items[0], items[1], quant_ord)
	fp.close()
		
# generate revenue for selected items		
def create_return(price_file, sel_file):
	fp = open(price_file, "r")
	pr_dict = {}
	for line in fp:
		items = line.split(",")
		it_pr = (items[0], items[1])
		pr_dict[it_pr] = int(items[2])
	fp.close()
	
	fs = open(sel_file, "r")
	for line in fs:
		line = line.rstrip() 
		items = line.split(",")
		it_pr = (items[0], items[1])
		rev = pr_dict[it_pr]
		rng = randrange(4, 8)
		rev =  randrange((rev * (100 - rng)) / 100, (rev * (100 + rng)) / 100) 
		print "%s,%s,%d" %(items[0], items[1], rev)
	fs.close()

# creates group item count and batch size
def create_count(price_file, batch_size):
	fp = open(price_file, "r")
	pr_dict = {}
	for line in fp:
		items = line.split(",")
		pr_dict[items[0]] = 1 + pr_dict.setdefault(items[0], 0)
	fp.close()
	
	for group in pr_dict:
		print "%s,%d,%d" %(group, pr_dict[group], batch_size)
		

op = sys.argv[1]
if (op == "price"):
	prod_count = int(sys.argv[2])
	stat_file = sys.argv[3]
	create_price(prod_count, stat_file)
elif (op == "inret"):
	price_file = sys.argv[2]
	quant_ord = int(sys.argv[3])
	create_init_return(price_file, quant_ord)
elif (op == "return"):
	price_file = sys.argv[2]
	sel_file = sys.argv[3]
	create_return(price_file, sel_file)
elif (op == "count"):
	price_file = sys.argv[2]
	batch_size = int(sys.argv[3])
	create_count(price_file, batch_size)
else:
	print "unknown command"	
	
