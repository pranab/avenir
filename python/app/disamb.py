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
emailDoms = ["yahoo.com", "gmail.com", "hotmail.com", "aol.com"]

def mutStr(st):
	"""
	mutate a char in string
	"""
	l = len(st)
	ci = randomInt(0, l - 1)
	cv = st[ci]
	if cv.isdigit():
		r = selectRandomFromList(dig)
	elif cv.isupper():
		r = selectRandomFromList(ucc)
	else:
		r = selectRandomFromList(lcc)
	
	nst = st[:ci] + r + st[ci+1:] if l > 1 else r
	return nst

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
			nc[1] = mutStr(nc[1])
			nfv = nc[0] + " "  + nc[1]
	elif fi == 1:
		#address
		mutated = False
		if isEventSampled(50):
			mutated = True
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
				mutated = False

		if not mutated:
			si = randomInt(0, 1)
			nc[si] = mutStr(nc[si])
		nfv = " ".join(nc)
	
	elif fi == 2:
		#city
		si = randomInt(0, le - 1) if le > 1 else 0
		nc[si] = mutStr(nc[si])
		nfv = " ".join(nc) if le > 1 else nc[0]
	
	elif fi == 3:
		#state
		nc[0] = mutStr(nc[0])
		nfv = nc[0]
		
	elif fi == 4:
		#zip
		nc[0] = mutStr(nc[0])
		nfv = nc[0]

	elif fi == 5:
		#email
		if isEventSampled(60):
			nc[0] = mutStr(nc[0])
			nfv = nc[0]
		else:
			nfv = genLowCaseID(randomInt(4, 10)) + "@" + selectRandomFromList(emailDoms)
	
	mrec[fi] = nfv
	return  mrec

def printNgramVec(ngv):
	print("ngram vector")
	for i in range(len(ngv)):
		if ngv[i] > 0:
			print("{} {}".format(i, ngv[i]))
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
				z = rec[-5]
				if len(z) == 7:
					z = z[1:-1]
				nrec.append(z)
				nrec.append(rec[-2][1:-1])
				print(",".join(nrec))
			i += 1
			
	elif op == "genpn":
		""" generate pos pos and pos neg paire """
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

	elif op == "sim":
		srcFilePath = sys.argv[2]
		cng = CharNGram(["lcc", "ucc", "dig"], 3, True)
		spc = ["@", "#", "_", "-", "."]
		cng.addSpChar(spc)
		cng.setWsRepl("$")
		c = 0
		for rec in fileRecGen(srcFilePath, ","):
			#print(",".join(rec))
			sim = list()
			for i in range(6):
				#print("field " + str(i))
				if i == 3:
					s = levenshteinSimilarity(rec[i],rec[i+6])
				else:
					ngv1 = cng.toMgramCount(rec[i])
					ngv2 = cng.toMgramCount(rec[i+6])
					#printNgramVec(ngv1)
					#printNgramVec(ngv2)
					s = cosineSimilarity(ngv1, ngv2)
				sim.append(s)
			ss = toStrFromList(sim, 6)
			print(ss + "," + rec[-1])
			c += 1
				
	elif op == "nnTrain":
		"""
		tran neural network model
		"""
		prFile = sys.argv[2]
		regr = FeedForwardNetwork(prFile)
		regr.buildModel()
		FeedForwardNetwork.batchTrain(regr)
				
	
