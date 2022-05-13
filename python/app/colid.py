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
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../supv"))
sys.path.append(os.path.abspath("../mlextra"))

from mlutil import *
from util import *
from sampler import *
from tnn import *
from daexp import *

"""
column type idenification
"""
def getStats(expl, ds):
	"""
	col stats
	"""
	stats = expl.getCatAlphaCharCountStats(ds)
	acmean = stats["mean"]
	acsd = stats["std dev"]
	
	stats = expl.getCatNumCharCountStats(ds)
	ncmean = stats["mean"]
	ncsd = stats["std dev"]
	
	stats = expl.getCatFldLenStats(ds)
	lmean = stats["mean"]
	lsd = stats["std dev"]
	
	stats = expl.getCatCharCountStats(ds, ' ')
	spmean = stats["mean"]
	spsd = 	 stats["std dev"]
	
	return [lmean, lsd, acmean, acsd, ncmean, ncsd, spmean, spsd]	
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--fpath', type=str, default = "none", help = "source file path")
	parser.add_argument('--ssize', type=int, default = 50, help = "sample size")
	parser.add_argument('--nsamp', type=int, default = 100, help = "number of samples")
	parser.add_argument('--sfpref', type=str, default = "none", help = "sample filename prefix")
	parser.add_argument('--fncnt', type=int, default = 1000, help = "file name counter")
	parser.add_argument('--sfdir', type=str, default = "none", help = "sample file directory")
	parser.add_argument('--cfpath', type=str, default = "none", help = "column features file path")
	parser.add_argument('--npair', type=int, default = 2, help = "positive and negative no of pairs")
	args = parser.parse_args()
	op = args.op
	
	if op == "sample":
		""" generate sample files """
		fpath = args.fpath
		tdata = getFileLines(fpath)
		for i in range(args.nsamp):
			sfpath = args.sfdir + "/" +  args.sfpref + "-" + str(args.fncnt + i) + ".txt"
			with open(sfpath, "w") as fh:
				for j in range(args.ssize):
					rec = selectRandomFromList(tdata)[:3]
					mfield = randint(0,2)
					for k in range(3):
						if k != mfield:
							fv = selectRandomFromList(tdata)[k]
							rec[k] = fv
					fh.write(",".join(rec) + "\n")
	
	elif op == "cfeatures":
		""" create column features """
		sfpaths = getAllFiles(args.sfdir)
		expl = DataExplorer(False)
		for sfp in 	sfpaths:
			names = list()
			addressses = list()
			cities = list()
			for rec in fileRecGen(sfp):
				names.append(rec[0])
				addressses.append(rec[1])
				cities.append(rec[2])
			
			expl.addListCatData(names, "name")		
			expl.addListCatData(addressses, "address")		
			expl.addListCatData(cities, "city")		
			
			stats = getStats(expl, "name")
			print(toStrFromList(stats, 3) + ",N")
			stats = getStats(expl, "address")
			print(toStrFromList(stats, 3) + ",A")
			stats = getStats(expl, "city")
			print(toStrFromList(stats, 3) + ",C")
	
	elif op == "cpairs":
		""" create col features pair """
		tdata = getFileLines(args.cfpath)
		le = len(tdata)
		for i in range(le):
			r1 = tdata[i]
			cp, cn = args.npair, args.npair
			while cp > 0 or cn > 0:
				j = randint(0, le - 1)
				while j == i:
					j = randint(0, le - 1)
				r2 = tdata[j]
				if r1[-1] == r2[-1]:
					#same type
					if cp > 0:
						#print(i,j,cp,cn)
						#print("P and P")
						r = r1[:-1].copy()
						r.extend(r2[:-1].copy())
						r.append("1")
						cp -= 1
						print(",".join(r))
				else:
					#opposite type
					if cn > 0:
						#print(i,j,cp,cn)
						#print("P and N")
						r = r1[:-1].copy()
						r.extend(r2[:-1].copy())
						r.append("0")
						cn -= 1
						print(",".join(r))
					