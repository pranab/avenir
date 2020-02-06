#!/usr/bin/python

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
import math
from datetime import datetime
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

"""
Time series generation for general time series with optional trend, cycles and random remainder 
components and random walk time series 
"""

def loadConfig(configFile):
	"""
	load config file
	"""
	defValues = {}
	defValues["window.size"] = (None, "missing time window size")
	defValues["window.samp.interval"] = ("fixed", None)
	defValues["window.samp.interval.params"] = (None, "missing time interval parameters")
	defValues["window.samp.align.unit"] = (None, None)
	defValues["output.value.type"] = ("float", None)
	defValues["output.value.precision"] = (3, None)
	defValues["output.time.format"] = ("epoch", None)
	defValues["ts.base"] = ("mean", None)
	defValues["ts.base.params"] = (None, "missing time series base parameters")
	defValues["ts.trend"] = ("nothing", None)
	defValues["ts.trend.params"] = (None, None)
	defValues["ts.cycles"] = ("nothing", None)
	defValues["ts.cycle.year.params"] = (None, None)
	defValues["ts.cycle.week.params"] = (None, None)
	defValues["ts.cycle.day.params"] = (None, None)
	defValues["ts.random"] = (True, None)
	defValues["ts.random.params"] = (None, None)
	defValues["rw.init.value"] = (5.0, None)
	defValues["rw.range"] = (1.0, None)
	defValues["corr.file.path"] = (None, None)
	defValues["corr.file.col"] = (None, None)
	defValues["corr.noise.stddev"] = (None, None)
	

	config = Configuration(configFile, defValues)
	return config

def getDateTime(tm, tmFormat):
	"""
	returns either epoch time for formatted date time
	"""
	if tmFormat == "epoch":
		dt = tm
	else:
		dt = datetime.fromtimestamp(tm)
		dt = dt.strftime("%Y-%m-%d %H:%M:%S")
	return dt


if __name__ == "__main__":
	op = sys.argv[1]
	confFile = sys.argv[2]
	config = loadConfig(confFile)
	delim = ","
	
	winSz = config.getStringConfig("window.size")[0]
	items = winSz.split("_")
	curTm, pastTm = pastTime(int(items[0]), items[1])
	
	sampIntvType = config.getStringConfig("window.samp.interval")[0]
	sampIntv = config.getStringConfig("window.samp.interval.params")[0].split(delim)
	intvDistr = None
	if sampIntvType == "fixed":
		sampIntv = int(sampIntv[0])
	elif sampIntvType == "random":
		siMean = float(sampIntv[0])
		siSd = float(sampIntv[1])
		intvDistr = GaussianRejectSampler(siMean,siSd)
	else:
		raise ValueError("invalid sampling interval type")
		
	sampAlignUnit = config.getStringConfig("window.samp.align.unit")[0]
	
	#output
	tsValType = config.getStringConfig("output.value.type")[0]
	valPrecision = config.getIntConfig("output.value.precision")[0]
	tsTimeFormat = config.getStringConfig("output.time.format")[0]
	if tsValType == "int":
		ouForm = "{},{}"
	else:
		ouForm = "{},{:."  + str(valPrecision) + "f}"
	
	
	#generic time series 
	if op == "gen":
		tsBaseType = config.getStringConfig("ts.base")[0]
		items = config.getStringConfig("ts.base.params")[0].split(delim)
		if tsBaseType == "mean":
			tsMean = float(items[0])
		elif tsBaseType == "exp":
			tsAlpha = float(items[0])
			tsHist = toFloatList(items[1:])
		else:
			raise ValueError("invalid base type")
		
		tsTrendType = config.getStringConfig("ts.trend")[0]
		items = config.getStringConfig("ts.trend.params")[0].split(delim)
		if tsTrendType == "linear":
			tsTrendSlope = float(items[0])
		elif tsTrendType == "quadratic":
			tsTrendQuadParams = toFloatList(items)
		elif tsTrendType == "logistic":
			tsTrendLogParams = toFloatList(items)
		else:
			raise ValueError("invalid trend type")
		
		cycles = config.getStringConfig("ts.cycles")[0].split(delim)
		yearCycle = weekCycle = dayCycle = None
		for c in cycles:
			key = "ts.cycle." + c + ".params"
			cycleValues = config.getStringConfig(key)[0].split(delim)
			if c == "year":
				yearCycle = toFloatList(cycleValues)
			elif c == "week":
				weekCycle = toFloatList(cycleValues)
			elif c == "day":
				dayCycle = toFloatList(cycleValues)
			
		
		tsRandom = config.getBooleanConfig("ts.random")[0]
		if tsRandom:
			items = config.getStringConfig("ts.random.params")[0].split(delim)
			tsRandMean = float(items[0])
			tsRandStdDev = float(items[1])
			tsRandDistr = GaussianRejectSampler(tsRandMean,tsRandStdDev)
		
	
		if sampAlignUnit:
			pastTm = timeAlign(pastTm, sampAlignUnit)	
	
		if intvDistr:
			sampIntv = int(intvDistr.sample())
	
		sampTm = pastTm
		counter = 0

		while (sampTm < curTm):
			curVal = 0
		
			#base
			if tsBaseType == "mean":
				curVal = tsMean
			else:
				alphaInv = 1
				for i in reversed(range(lenn(tsHist))):
					curVal = tsAplha * alphaInv * tsHist[i] 
					alphaInv *= (1.0 - tsAplha)
		
			#trend
			if tsTrendType == "linear":
				curVal += counter * tsTrendSlope
			elif tsTrendType == "quadratic":
				curVal += tsTrendQuadParams[0] * counter + tsTrendQuadParams[1] * counter * counter
			elif tsTrendType == "logistic":
				ex = math.exp(-tsTrendLogParams[0] * counter)
				curVal += tsTrendLogParams[0] * (1.0 - ex) / (1.0 + ex)
			counter += 1
		
			#cycle
			if yearCycle:
				month = monthOfYear(sampTm)
				curVal += yearCycle[month]
			if weekCycle:
				day = dayOfWeek(sampTm)
				curVal += weekCycle[day]
			if dayCycle:
				hour = hourOfDay(sampTm)
				curVal += dayCycle[hour]

			#random remainder
			if tsRandStdDev:
				curVal += tsRandDistr.sample()
	
			#date time
			if tsTimeFormat == "epoch":
				dt = sampTm
			else:
				dt = datetime.fromtimestamp(sampTm)
				dt = dt.strftime("%Y-%m-%d %H:%M:%S")
		
			#update history
			if tsBaseType == "exp":
				tsHist.append(curVal)
				tsHist.pop(0)
		
			#value
			if tsValType == "int":
				curVal = int(curVal)

			print(ouForm.format(dt, curVal))
	
			#next
			if intvDistr:
				sampIntv = int(intvDistr.sample())
			sampTm += sampIntv
	
	#random walk time series		
	elif op == "rw":
		initVal = config.getFloatConfig("rw.init.value")[0]
		ranRange = config.getFloatConfig("rw.range")[0]
		sampTm = pastTm
		curVal = initVal
		
		while (sampTm < curTm):
			#value
			if tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = getDateTime(sampTm, tsTimeFormat)
			
			print(ouForm.format(dt, curVal))
			
			#next
			curVal += randomFloat(-ranRange, ranRange)
			
			if intvDistr:
				sampIntv = int(intvDistr.sample())
			sampTm += sampIntv
	
	# generates correlated time seriec
	elif op == "corr":
		refFile = config.getStringConfig("corr.file.path")[0]
		refCol = config.getIntConfig("corr.file.col")[0]
		noiseSd = config.getFloatConfig("corr.noise.stddev")[0]
		noiseDistr = GaussianRejectSampler(0,noiseSd)
		for rec in fileRecGen(refFile, ","):
			val = float(rec[refCol]) + noiseDistr.sample()
			val = "%.3f" %(val)
			rec[refCol] = val
			nRec = ",".join(rec)
			print(nRec)

		
	else:
		raise ValueError("ivalid time series type")
			
	

