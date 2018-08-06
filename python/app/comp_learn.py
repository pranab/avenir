#!/usr/bin/python

import os
import sys
from random import randint
import time
import math
sys.path.append(os.path.abspath("../lib"))
from util import *

def numSamples(numHyp, errors, prThresholds):
	for e in errors:
		for p in prThresholds:
			m = math.log(numHyp / p) / e
			m = long(m)
			print "%d,%.3f,%.3f,%d" %(numHyp,e,p,m)

def numSamplesWithLn(numHypLn, errors, prThresholds):
	for e in errors:
		for p in prThresholds:
			m = (numHypLn + math.log(1/p)) / e
			m = long(m)
			print "%.3f,%.3f,%d" %(e,p,m)

#conjuction of all feature variables
def termsHypSpace(featureAttrCard, classAttrCard):
	print "all terms:"
	numHyp = 1
	for f in featureAttrCard:
		numHyp = numHyp * (f + 1)
	numHyp = numHyp * classAttrCard
	return numHyp

# k term dnf : 	
def disjunctiveHypSpace(featureAttrCard, classAttrCard, cSize, dSize):
	print "k term dnf:"
	numHyp = numValueCombinations(featureAttrCard, cSize)
		
	n = 1
	for i in range(0, dSize):
		n = n * (numHyp - i)
	
	f = 1
	for i in range(0,dSize):
		f = f * (dSize -i)
		
	numHyp = n / f
	
	numHyp = numHyp * classAttrCard
	return numHyp

# k cnf : disjuctions of size dSize then all possible conjunctions
def conjunctiveHypSpace(featureAttrCard, classAttrCard, dSize):
	print "k cnf:"
	numHyp = numValueCombinations(featureAttrCard, dSize)
	e = math.exp(1)
	numHypLn = numHyp / math.log(e,2)  + math.log(classAttrCard)
	return numHypLn

def numValueCombinations(featureAttrCard, numVars):
	numFeature = len(featureAttrCard)
	numHyp = 0
	if (numVars == 3):
		for i in range(0, numFeature):
			for j in range(1, numFeature):
				for k in range(2, numFeature):
					numHyp = numHyp + featureAttrCard[i] * featureAttrCard[j] * featureAttrCard[k]  	
	elif (numVars == 4):
		for i in range(0, numFeature):
			for j in range(1, numFeature):
				for k in range(2, numFeature):
					for l in range(3, numFeature):
						numHyp = numHyp + featureAttrCard[i] * featureAttrCard[j] * featureAttrCard[k] * featureAttrCard[l] 	
	elif (numVars == len(featureAttrCard)):
		numHyp = 1
		for f in featureAttrCard:
			numHyp = numHyp * f 		
	return numHyp
	
#####################################################################################	
featureAttrCard = []
items = sys.argv[1].split(",")
for item in items:
	featureAttrCard.append(int(item))	

classAttrCard = int(sys.argv[2])
hypSpace =  sys.argv[3]
if (hypSpace == "terms"):
	numHyp = termsHypSpace(featureAttrCard, classAttrCard)
elif (hypSpace == "dnf"):
	if (len(sys.argv) == 6):
		cSize = int(sys.argv[4])
		dSize = int(sys.argv[5])
	else:
		cSize = len(featureAttrCard)
		dSize = int(sys.argv[4])
	numHyp = disjunctiveHypSpace(featureAttrCard, classAttrCard, cSize, dSize)
elif (hypSpace == "cnf"):	
	dSize = int(sys.argv[4])
	numHypLn = conjunctiveHypSpace(featureAttrCard, classAttrCard, dSize)
	
	
errors = [0.01, 0.02, 0.03, 0.04, 0.05]
prThresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

if (hypSpace == "cnf"):
	numSamplesWithLn(numHypLn, errors, prThresholds)
else:
	numSamples(numHyp, errors, prThresholds)
