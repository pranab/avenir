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
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import random
import jprops
import statistics as stat
from matplotlib import pyplot
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *
from stats import *

"""
human bias detection
"""
class BiasDetector(object):
	"""
	bias detection
	"""
	def __init__(self, fpath, ftypes):
		self.fpath = fpath
		self.ftypes = ftypes
	
	def extLift(self, pfe, cl, fe = None):
		"""
		exyended lift
		"""
		allCnt = 0
		feCnt = 0
		feClCnt = 0
		afeCnt = 0
		afeClCnt = 0
		for rec in fileTypedRecGen(self.fpath, self.ftypes):
			feMatched = True if fe is None else self.__isMatched(rec, fe)
			if feMatched:
				feCnt += 1
				clMatched = self.__isMatched(rec, cl)
				if clMatched:
					feClCnt += 1
				if self.__isMatched(rec, pfe):
					afeCnt += 1
					if 	clMatched:
						afeClCnt += 1
			allCnt += 1
		
		print("feCnt {}   feClCnt {}   afeCnt {}   afeClCnt {}".format(feCnt, feClCnt, afeCnt, afeClCnt))
		#unprotected feature and all feature confidence	
		feConf = feClCnt / feCnt
		afeConf = afeClCnt / afeCnt
		elift = afeConf / feConf
		res = createMap("normal featurre conf", feConf, "all feature conf", afeConf, "extended lift", elift)
		return res
			
	def contrLift(self, pfe, cl, fe = None):
		"""
		contrasted lift
		"""
		allCnt = 0
		afeCntOne = 0
		afeClCntOne = 0
		afeCntTwo = 0
		afeClCntTwo = 0
		pfeOne = pfe[:2]
		pfeTwo = list()
		pfeTwo.append(pfe[0])
		pfeTwo.append(pfe[2])
		for rec in fileTypedRecGen(self.fpath, self.ftypes):
			feMatched = True if fe is None else self.__isMatched(rec, fe)
			if feMatched:
				clMatched = self.__isMatched(rec, cl)
				(afeCntOne, afeClCntOne) = self.__protFeatMatchCount(rec, pfeOne, clMatched, afeCntOne, afeClCntOne)
				(afeCntTwo, afeClCntTwo) = self.__protFeatMatchCount(rec, pfeTwo, clMatched, afeCntTwo, afeClCntTwo)
			allCnt += 1
			
		afeConfOne = afeClCntOne / afeCntOne
		afeConfTwo = afeClCntTwo / afeCntTwo
		clift = afeConfOne / afeConfTwo
		res = createMap("protected featurre value one conf", afeConfOne, "protected featurre value two conf", afeConfTwo, 
		"contrasted lift", clift)
		return res
	
	def odds(self, pfe, cl, fe = None):
		"""
		odds ratio
		"""
		allCnt = 0
		afeCnt = 0
		afeClCnt = 0
		for rec in fileTypedRecGen(self.fpath, self.ftypes):
			feMatched = True if fe is None else self.__isMatched(rec, fe)
			if feMatched:
				clMatched = self.__isMatched(rec, cl)
				if self.__isMatched(rec, pfe):
					afeCnt += 1
					if 	clMatched:
						afeClCnt += 1
			allCnt += 1
		
		#unprotected feature and all feature confidence	
		afeConf = afeClCnt / afeCnt
		odds = afeConf / (1.0 - afeConf)
		res = createMap("all feature conf", afeConf, "odds", odds)
		return res

	def olift(self, pfe, cl, fe = None):
		"""
		odds lift
		"""
		pfeOne = pfe[:2]
		pfeTwo = list()
		pfeTwo.append(pfe[0])
		pfeTwo.append(pfe[2])
		oddsOne = self.odds(pfeOne, cl, fe)["odds"]
		oddsTwo = self.odds(pfeTwo, cl, fe)["odds"]
		olift = oddsOne / oddsTwo
		res = createMap("odds one", oddsOne, "odds two", oddsTwo, "odds lift", olift)
		return res
		

	def statParity(self, pfe, cl):
		"""
		exyended lift
		"""
		allCnt = 0
		clCnt = 0
		pfeCnt = 0
		pfeClCnt = 0
		for rec in fileTypedRecGen(self.fpath, self.ftypes):
			allCnt += 1
			pfeMatched = self.__isMatched(rec, pfe)
			if pfeMatched:
				pfeCnt += 1
			if self.__isMatched(rec, cl):
				clCnt += 1
				if pfeMatched:
					pfeClCnt += 1
		etarget = clCnt / allCnt
		ebtarget = pfeClCnt / pfeCnt
		
		parity = ebtarget / etarget
		res = createMap("statistical parity", parity)
		return res				
		
	def __protFeatMatchCount(self, rec, pfe, clMatched, afeCnt, afeClCnt):
		"""
		protected feature match
		"""
		if self.__isMatched(rec, pfe):
			afeCnt += 1
			if 	clMatched:
				afeClCnt +=1
		r = (afeCnt, afeClCnt)
		return r
			
	def __isMatched(self, rec, values):
		"""
		coolumn wise value match
		"""
		matched = True
		for i in range(0, len(values), 2):
			ci = values[i]
			cv = values[i+1]
			matched = matched and rec[ci] == cv
			if not matched:
				break
		return matched
	
			
			
	
