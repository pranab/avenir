#!/usr/bin/python

import sys
from random import randint
#hours: 1, 2, 3
#recommendations: (C)ross sell, (S)ocial, (N)one

numRetarget = int(sys.argv[1])
conversion = {'1C':75, '1S':60, '1N':50, '2C':60, '2S':40, '2N':30, '3C':20, '3S':20, '3N':15}
types = ['1C', '1S', '1N', '2C', '2S', '2N', '3C', '3S', '3N']

for i in range(1,numRetarget):
	custID = 1000000 + randint(0, 999999)
	type = types[randint(0,8)]
	convProb = conversion[type]
	c = randint(1,100)
	if(c < convProb):
		conv =  'Y'
	else:
		conv = 'N'
	amount = 20 + randint(0,300)
	print "%d,%s,%d,%s"%(custID,type,amount,conv)
	
		