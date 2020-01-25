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
from statsmodels.stats.stattools import jarque_bera
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
	defValues["diff.order"] = (1, None)
	defValues["acf.lags"] = (40, None)
	defValues["acf.alpha"] = (None, None)
	defValues["acf.diff"] = (False, None)
	defValues["pacf.lags"] = (40, None)
	defValues["pacf.alpha"] = (None, None)
	defValues["adf.regression"] = ("c", None)
	defValues["adf.autolag"] = ("AIC", None)
	defValues["kpss.regression"] = ("c", None)
	defValues["hist.cumulative"] = (False, None)
	defValues["hist.density"] = (False, None)
	
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
	
	#plot data
	if op == "draw":
		pyplot.plot(data)
		pyplot.show()
	
	#difference
	elif op == "diff":
		order = config.getIntConfig("diff.order")[0]
		for i in range(order):
			data = difference(data)
		pyplot.plot(data)
		pyplot.show()

	#auto correlation
	elif op == "acf":
		#print(data)
		diff = config.getBooleanConfig("acf.diff")[0]
		if diff:
			data = difference(data)
			pyplot.plot(data)
			pyplot.show()
			
		lags = config.getIntConfig("acf.lags")[0]
		alpha =config.getFloatConfig("acf.alpha")[0]
		tsaplots.plot_acf(data, lags = lags, alpha = alpha)
		pyplot.show()

	#partial auto correlation
	elif op == "pacf":
		lags = config.getIntConfig("pacf.lags")[0]
		alpha =config.getFloatConfig("pacf.alpha")[0]
		tsaplots.plot_pacf(data, lags = lags, alpha = alpha)
		pyplot.show()
	
	#stationarity test
	elif op == "adf":
		regression = config.getStringConfig("adf.regression")[0]
		autolag = config.getStringConfig("adf.autolag")[0]
		result = adfuller(data, regression=regression, autolag=autolag)
		print(f'ADF Statistic: {result[0]}')
		print(f'p value: {result[1]}')
		print(f'num lags: {result[2]}')
		print(f'num observation for regression: {result[3]}')
		print('Critial Values:')
		for key, value in result[4].items():
			print(f'   {key}, {value}')    
			
	#stationarity test
	elif op =="kpss":
		regression = config.getStringConfig("kpss.regression")[0]
		statistic, pvalue, nLags, criticalValues = kpss(data, regression=regression)
		print(f'KPSS Statistic: {statistic}')
		print(f'pvalue: {pvalue}')
		print(f'num lags: {nLags}')
		print('Critial Values:')
		for key, value in criticalValues.items():
			print(f'   {key} : {value}')	
				
	#normalcy test	
	elif op == "jarqBera":	
		jb, jbpv, skew, kurtosis =  jarque_bera(data)
		print(f'pvalue: {jbpv}')
		print(f'skew: {skew}')
		print(f'kurtosis: {kurtosis}')
		
	#histogram
	elif op == "hist":	
		cumulative = config.getBooleanConfig("hist.cumulative")[0]
		density = config.getBooleanConfig("hist.density")[0]
		pyplot.hist(data, cumulative=cumulative, density=density)
		pyplot.show()

		
	else:
		raise ValueError("unknown command")
		
		