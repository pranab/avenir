#!/usr/bin/python

import os
import sys
from random import randint
import time
from compiler.ast import flatten
sys.path.append(os.path.abspath("../lib"))
from util import *

itemCount = int(sys.argv[1])
tripletCount = int(sys.argv[2])
xactionCount = int(sys.argv[3])

items = []
triplets = []
pairs = []
singles = []

for i in range(0,itemCount):
	it = genID(10)
	items.append(it)

#generate triplets
for i in range(0,tripletCount):
	triplet = []	
	for j in range(0,3):
		triplet.append(selectRandomFromList(items))
	triplets.append(triplet)

#generate pairs
for triplet in triplets:
	for i in range(0,3):
		for j in range(i+1,3):
			pair = []
			pair.append(triplet[i])
			pair.append(triplet[j])	
			pairs.append(pair)

#additional frequent pairs
for i in range(0,5):
	pair = []
	pair.append(selectRandomFromList(items))
	pair.append(selectRandomFromList(items))	
	pairs.append(pair)

#generate frquent singles
for pair in pairs:
	singles.append(pair[0])
	singles.append(pair[1])

#additional frequent singles
for i in range(0,10):
	singles.append(selectRandomFromList(items))


#transactions
curTime = int(time.time())
day = 24 * 60 * 60
xactionTime = curTime - 30 * day

for i in range(0,xactionCount):
	xactionID = genID(12)
	xactionTime = xactionTime + randint(10,300)
	xactionItems = []
	r = randint(0,100)
	count = 0
	if (r < 40):
		#triplet
		triplet = selectRandomFromList(triplets)
		xactionItems.append(triplet)
		count = 3
	elif (r < 50):
		#pair
		pair = selectRandomFromList(pairs)
		xactionItems.append(pair)
		count = 2
	elif (r < 60):
		#single
		single = selectRandomFromList(singles)
		xactionItems.append(single)
		count = 1

	xactionCount = 3 + randint(0,10)
	remaining = xactionCount - count
	for j in range(0, remaining):
		xactionItems.append(selectRandomFromList(items))

	flatXactionItems = ",".join(flatten(xactionItems))
	print "%s,%d,%s" %(xactionID, xactionTime, flatXactionItems)

	


