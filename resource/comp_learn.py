#!/usr/bin/python

import os
import sys
from random import randint
import time
import math
sys.path.append(os.path.abspath("../lib"))
from util import *

def numSamples(featureAttrCard, classAttrCard, errors, prThresholds):
	numHyp = 1
	for f in featureAttrCard:
		numHyp = numHyp * (f + 1)
	numHyp = numHyp * classAttrCard

	for e in errors:
		for p in prThresholds:
			m = math.log(numHyp / p) / e
			m = int(m)
			print "%.3f,%.3f,%d" %(e,p,m)


featureAttrCard = []
items = sys.argv[1].split(",")
for item in items:
	featureAttrCard.append(int(item))	

classAttrCard = int(sys.argv[2])
errors = [0.01, 0.02, 0.03, 0.04, 0.05]
prThreshold = [0.01, 0.02, 0.03, 0.04, 0.05]

numSamples(featureAttrCard, classAttrCard, errors, prThreshold)