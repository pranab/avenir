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
from datetime import datetime
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from matplotlib import pyplot
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

def loadConfig(configFile):
	"""
	load config file
	"""
	defValues = {}
	defValues["data.filePath"] = (None, "missing file path")
	defValues["data.col.index"] = (None, "missing col index")
	defValues["data.row.range"] = ("all", None)
	defValues["acf.lags"] = (40, None)
	defValues["acf.alpha"] = (None, None)
	defValues["acf.diff"] = (False, None)
	
	config = Configuration(configFile, defValues)
	return config

def loadData(config):
	"""
	loads a column
	"""
	filePath = config.getStringConfig("data.filePath")[0]
	col = config.getIntConfig("data.col.index")[0]
	data = getFileColumnAsFloat(filePath ,col,  ",")
	if not config.getStringConfig("data.row.range")[0] == "all":
		ra = config.getIntListConfig("data.row.range", ",")[0]
		data = data[ra[0]:ra[1]]
	return np.array(data)


if __name__ == "__main__":
	op = sys.argv[1]
	confFile = sys.argv[2]
	config = loadConfig(confFile)
	data = loadData(config)
	
	if op == "draw":
		pyplot.plot(data)
		pyplot.show()
	
	elif op == "acf":
		#print(data)
		diff = config.getBooleanConfig("acf.diff")[0]
		if diff:
			data = difference(data)
			pyplot.plot(data)
			pyplot.show()
			
		lags = config.getIntConfig("acf.lags")[0]
		alpha =config.getFloatConfig("acf.alpha")[0]
		tsaplots.plot_acf(np.array(data), lags = lags, alpha = alpha)
		pyplot.show()
	
	elif op == "adf":
		result = adfuller(data, autolag='AIC')
		print(f'ADF Statistic: {result[0]}')
		print(f'p value: {result[1]}')
		print(f'num lags: {result[2]}')
		print(f'num observation for regression: {result[3]}')
		for key, value in result[4].items():
			print('Critial Values:')
			print(f'   {key}, {value}')    
			
	elif op =="kpss":
		statistic, pvalue, nLags, criticalValues = kpss(data)
		print(f'KPSS Statistic: {statistic}')
		print(f'pvalue: {pvalue}')
		print(f'num lags: {nLags}')
		print('Critial Values:')
		for key, value in criticalValues.items():
			print(f'   {key} : {value}')		
		
		
	else:
		raise ValueError("unknown command")
		
		