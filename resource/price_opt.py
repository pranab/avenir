#!/usr/bin/python

import sys
from random import randrange


def create_price(prod_count):
	for pd in range(1, prod_count):
		prod_id = randrange(1000000, 8000000)
		num_price = randrange(6,12)
		price_delta = randrange(2,4)
		price = randrange(10,80)
		rev = randrange(10000, 30000)
		rev_delta = randrange(500, 1500)
		half_way =  num_price / 2 + randrange(-2,2)
		for pr in range(1, num_price):
			print "%d,%d,%d" %(prod_id, price, rev)
			price += price_delta
			if (pr < half_way):
				rev += rev_delta + randrange(-20,20)
			else:
				rev -= rev_delta + randrange(-20,20)

def create_return(price_file, sel_file):
	fp = open(price_file, "r")
	pr_dict = {}
	for line in fp:
		items = line.split(",")
		it_pr = (items[0], int(items[1]))
		pr_dict[it_pr] = int(items[2])
	fp.close()
	
	fs = open(sel_file, "r")
	for line in fs:
		line = line.rstrip() 
		items = line.split(",")
		it_pr = (items[0], int(items[1]))
		rev = pr_dict[it_pr]
		rng = randrange(4, 8)
		rev =  randrange((rev * (100 - rng)) / 100, (rev * (100 + rng)) / 100) 
		print "%s,%s,%d" %(items[0], items[1], rev)
	fs.close()

op = sys.argv[1]
if (op == "price"):
	prod_count = int(sys.argv[2])
	create_price(prod_count)
else:
	price_file = sys.argv[2]
	sel_file = sys.argv[3]
	create_return(price_file, sel_file)
	
	
