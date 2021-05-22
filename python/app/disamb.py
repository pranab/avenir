#!/usr/local/bin/python3

# beymani-python: Machine Learning
# Author: Pranab Ghosh
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# Package imports
import os
import sys
import random
import statistics 
import numpy as np
import matplotlib.pyplot as plt 
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../supv"))
sys.path.append(os.path.abspath("../text"))
from util import *
from sampler import *
from tnn import *
from txproc import *

"""
causality analysis 
"""

def mutStr(st):
	"""
	mutate a char in string
	"""
	l = len(st)
	if l > 1:
		ci = randomInt(0, l - 1)
		cv = st[ci]
		if isdigit(cv):
			st[ci] = selectRandomFromList(dig)
		elif isupper(cv):
			st[ci] = selectRandomFromList(ucc)
		else:
			st[ci] = selectRandomFromList(lcc)
		
	return st

def createPosMatch(rec, fi):
	"""
	create positive match by mutating a field
	"""
	mrec = rec.copy()
	fv = mrec[fi]
	nc = fv.split()
	le = len(nc)
	if fi == 0:
		#name
		if isEventSampled(50):
			nfv = nc[0] + " " +  selectRandomFromList(ucc) + " " +  nc[1]
		else:
			ci = randomInt(1, len(nc[1]))
			nc[1][ci] = selectRandomFromList(lcc)
			nfv = nc[0] + " "  + nc[1]
	elif fi == 1:
		#address
		if isEventSampled(50):
			s = nc[-1]
			if s == "Rd":
				nc[-1] = "Road"
			elif s == "Ave":
				nc[-1] = "Avenue"
			elif s == "St":
				nc[-1] = "Street"
			elif s == "Dr":
				nc[-1] = "Drive"
		else:
			if isEventSampled(50):
				mutStr(nc[0])
			else:
				si = selectRandomFromList(1, len(nc)-2)
				mutStr(nc[si])
		nfv = " ".join(nc)
	
	elif fi == 2:
		#city
		si = selectRandomFromList(0, le - 1) if le > 1 else 0
		mutStr(nc[si])
		nfv = " ".join(nc) if le > 1 else nc[0]
	
	elif fi == 3:
		#state
		mutStr(nc[0])
		nfv = nc[0]
		
	elif fi == 4:
		#zip
		mutStr(nc[0])
		nfv = nc[0]

	elif fi == 5:
		#email
		mutStr(nc[0])
		nfv = nc[0]
	
	mrec[fi] = nfv
	return mrec

def createNegMatch(tdata, ri):
	"""
	create negative match by randomly selecting another record
	"""
	nri = randomInt(0, len(tdata)-1)
	while nri == ri:
		nri = randomInt(0, len(tdata)-1)
	return tdata[nri]
	
if __name__ == "__main__":
	op = sys.argv[1]
	if op == "gen":
		srcFilePath = sys.argv[2]
		i = 0
		for rec in fileRecGen(srcFilePath, ","):
			if i > 0:
				nrec = list()
				fname = rec[0][1:-1]
				lname = rec[1][1:-1]
				nrec.append(fname + " " + lname)
				nrec.append(rec[-9][1:-1])
				nrec.append(rec[-8][1:-1])
				nrec.append(rec[-6][1:-1])
				nrec.append(rec[-5])
				nrec.append(rec[-2][1:-1])
				print(",".join(nrec))
			i += 1
			
	elif op == "genpn":
		srcFilePath = sys.argv[2]
		tdata = getFileLines(srcFilePath)
		
		ri = 0
		for rec in fileRecGen(srcFilePath, ","):
			for _ in range(2):
				fi = randomInt(0, 5)
				mrec = createPosMatch(rec, fi)
				if isEventSampled(30):
					fi = randomInt(0, 5)
					mrec = createPosMatch(mrec, fi)
				print(",".join(rec) + "," + ",".join(mrec) + "," + "1")
				
			for _ in range(2):
				mrec = createNegMatch(tdata, ri)
				print(",".join(rec) + "," + mrec + "," + "0")
				
			ri += 1
