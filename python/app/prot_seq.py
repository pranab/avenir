#!/usr/bin/python

import os
import sys
from random import randint
import time
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *


op = sys.argv[1]
numSeq = int(sys.argv[2])
minLen = int(sys.argv[3])
maxLen = int(sys.argv[4])
mutPercent = int(sys.argv[5])

amAcid = ["A", "C", "D", "E", "F", "G", "H", "I", "K",  "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

def createSeed(sLen):
	residues = []
	for i in range(sLen):
		residues.append(selectRandomFromList(amAcid))
	return residues

def cloneSeq(seq, mutFrac):
	sizeChange = randint(1, 6)
	if (isEventSampled(50)):
		#extend
		nSeq = seq[:]
		for i in range(sizeChange):
			nSeq.append(selectRandomFromList(amAcid))
	else:
		#shrink
		nSeq = seq[:-sizeChange]
	
	#mutate
	maxMut = int(len(nSeq) * mutFrac)
	numMut = maxMut + randint(0, 10)
	for i in range(numMut):
		pos = randint(0, len(nSeq) - 1)
		nSeq[pos] = selectRandomFromList(amAcid)
	
	key = genID(12)
	seqStr = key + "," +  ":".join(nSeq)
	return seqStr
	
def createSeq(sLen):
	key = genID(12)
	residues = []
	for i in range(sLen):
		residues.append(selectRandomFromList(amAcid))
	seq = key + "," +  ":".join(residues)
	return seq
		
if op == "divergent":
	numSeed = int(numSeq * 0.1)
	seeds = []
	for i in range(numSeq + numSeed):
		if (i < numSeed):
			sLen = randint(minLen, maxLen)
			seq = createSeed(sLen)
			seeds.append(seq)
		else:
			seed = selectRandomFromList(seeds)
			seq = cloneSeq(seed, mutPercent / 100.0)
			print seq
		
		