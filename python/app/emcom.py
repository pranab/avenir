#!/usr/local/bin/python3

# avenir-python: Machine Learning
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

import os
import sys
from random import randint
import time
from array import *
import argparse
import statistics
import numpy as np
from sklearn.neighbors import KDTree
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../supv"))
sys.path.append(os.path.abspath("../mlextra"))

from mlutil import *
from util import *
from sampler import *
from tnn import *

"""
generates email communication data for discovering subject matter experts using GCN
"""


##########################################################################################
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--nemp', type=int, default = 2000, help = "number of employees")
	parser.add_argument('--nexp', type=int, default = 100, help = "number of experts")
	parser.add_argument('--njun', type=int, default = 300, help = "number of juniors")
	parser.add_argument('--nclust', type=int, default = 3, help = "number of communication clusters")
	parser.add_argument('--nsm', type=int, default = 3, help = "number of subject matters")
	parser.add_argument('--trsz', type=int, default = 3, help = "training data size")
	args = parser.parse_args()
	op = args.op
	
	if op == "gen":
		nclust = args.nclust
		nemp = args.nemp
		ndelta = int(.02 * nemp)
		nexp = args.nexp
		njun = args.njun
		noth = nemp - nexp - njun
		segCounts = [nexp, njun, noth]
		#print(segCounts)
		nsm = args.nsm
		
		#per cluster average
		avSegCounts = list()
		avSegCounts.append(int(nexp / nclust))
		avSegCounts.append(int(njun / nclust))
		avSegCounts.append(int(noth / nclust))
		perClustCount = int(nemp / nclust)
		prefix = ["E", "J", "O"]
		
		#message count distr
		msgCntDistr = [NormalSampler(500,30), NormalSampler(400, 30), NormalSampler(250, 15)]
		
		# msg length distr for each category received and sent
		msgLenDistr = [NormalSampler(200,20), NormalSampler(800, 50), NormalSampler(700, 50), NormalSampler(300,20), 
		NormalSampler(400,30), NormalSampler(400,30)]
		
		#bag of words vector for eacg subject matter
		tbow = list(map(lambda t : genAlmostUniformDistr(50,20), range(nsm)))
		shift = .5 / 50
		tbow = list(map(lambda d : mutDistr(d, shift, 10), tbow))
		gbow = genAlmostUniformDistr(50,5)
		
		avClSize = int(nemp / nclust)
		nClEdges = int(avClSize * (avClSize - 1) / 2)
		nEdges = int(nemp * (nemp - 1) / 2)
		nseg = 3
		clist = list(range(nclust))
		
		#generate Ids for all employee types in each cluster
		sme = dict()
		i = 0
		t = 0
		mint = 2 * nsm
		for c in range(nclust):
			for s in range(nseg):
				cnt = avSegCounts[s] + randomInt(-ndelta, ndelta)
				for _ in range(cnt):
					eid = prefix[s] + genID(9)
					
					#minimum topic allocation
					t = randomInt(0, nsm - 1) if i > mint else (t + 1) % nsm
					
					k = (c, s, t)
					appendKeyedList(sme, k, eid)
					i += 1
		#print("no of nodes " , i)	
				
		#generate nodes
		nidexes = dict()
		i = 0
		mask = list()
		nlabelled = 0
		for s in range(nseg):
			if s == 1:
				nlabelled = i
				r = (0, nlabelled)
				mask.append(r)
				remLabels = int((args.trsz - nlabelled) / 2)
			elif s == 2:
				beg =  mask[0][1]
				r = (beg, beg + remLabels)
				mask.append(r)
				nlabelled = i
				
			for c in range(nclust):
				for t in range(nsm):
					k = (c,s,t)
					eids = sme[k]
					for eid in eids:
						#msg count
						mcount = int(msgCntDistr[s].sample())
						
						#receved msg length
						rmlen = int(msgLenDistr[2 * s].sample())
						
						#sent msg length
						smlen = int(msgLenDistr[2 * s + 1].sample())
					
						if s < 2:
							#juniors and experts
							bow = tbow[t]
						else:
							#rest
							bow = gbow
							
						bow = mutDistr(bow, .1 / 50, 5)
						bow = toStrFromList(bow, 3)
						if s == 0:
							lab = t
						elif s == 1:
							lab = nsm
						else:
							lab = nsm + 1
						print("{},{},{},{},{},{}".format(eid, mcount, rmlen, smlen, bow, lab))
						appendKeyedList(nidexes, k, i)
						i += 1
		
		beg =  nlabelled
		r = (beg, beg + remLabels)
		mask.append(r)
		l = list(map(lambda r : str(r[0]) + ":" + str(r[1]), mask))
		m = " ".join(l)
		
		nnodes = i
		print("mask ", nnodes, m)
				
		#edges topic based between junior and experts
		e = 0
		for c in range(nclust):
			for t in range(nsm):
				k = (c,1,t)
				jeids = nidexes[k]
				k = (c,0,t)
				eeids = nidexes[k]
				leeids = len(eeids)
				for ji in jeids:
					if len(eeids) > 4:
						eis = selectRandomSubListFromList(eeids, randomInt(3,4))
					else:
						eis = eeids
					for ei in eis:
						print("{},{}".format(ji, ei))
						print("{},{}".format(ei, ji))
						e += 2
		#print("no of sme specific connections ", e)		
			
		#edges intra cluster
		e = 0
		for c in range(nclust):
			ne = int(nClEdges * randomFloat(.10, .15))
			for _  in range(ne):
				s = randomInt(0, nseg - 1)		
				t = randomInt(0, nsm - 1)
				k = (c,s,t)
				n1 = selectRandomFromList(nidexes[k])
				
				s = randomInt(0, nseg - 1)	
				t = randomInt(0, nsm - 1)
				k = (c,s,t)
				n2 = selectRandomFromList(nidexes[k])
				if n1 != n2:
					print("{},{}".format(n1, n2))
					print("{},{}".format(n2, n1))
					e += 2
		#print("no of intra cluster connections ", e)		
						
		#edges inter cluster
		e = 0
		ne = int(nEdges * randomFloat(.02, .04))
		for _  in range(ne):
			c = randomInt(0, nclust - 1)	
			s = randomInt(0, nseg - 1)	
			t = randomInt(0, nsm - 1)
			n1 = selectRandomFromList(nidexes[k])
			
			c = selectOtherRandomFromList(clist, c)	
			s = randomInt(0, nseg - 1)	
			t = randomInt(0, nsm - 1)
			n2 = selectRandomFromList(nidexes[k])
			print("{},{}".format(n1, n2))
			print("{},{}".format(n2, n1))
			e += 2
		#print("no of inter cluster connections ", e)		
				