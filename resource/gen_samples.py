#!/usr/bin/python

import os
import sys
from random import randint
import time
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

op = sys.argv[1]
numItems = int(sys.argv[2])

if (op == "genLatLong"):
	lat1 = float(sys.argv[3])
	long1 = float(sys.argv[4])
	lat2 = float(sys.argv[5])
	long2 = float(sys.argv[6])
	
	for i in range(numItems):
		(lat,long) = genLatLong(lat1, long1, lat2, long2)
		print "%.5f, %.5f" %(lat, long)
		
elif (op == "genId"):	
	len = int(sys.argv[3])
	for i in range(numItems):
		id = genID(len)
		print "%s" %(id)