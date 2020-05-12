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
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.stattools import jarque_bera
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from scipy.stats import ks_2samp
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import shapiro
from scipy.stats import kendalltau
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import chi2_contingency
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

"""
Various data exploration functions. Some are applicable for time series only.
https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/
https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
"""

def loadConfig(configFile):
	"""
	load config file
	"""
	defValues = {}
	defValues["data.filePath"] = (None, "missing file path")
	defValues["data.col.index"] = (None, "missing col index")
	defValues["data.row.range"] = ("all", None)
	defValues["data.filePath.extra"] = (None, None)
	defValues["data.col.index.extra"] = (None, None)
	defValues["data.row.range.extra"] = ("all", None)
	defValues["diff.order"] = (1, None)
	defValues["trend.remove"] = (False, None)
	defValues["acf.lags"] = (40, None)
	defValues["acf.alpha"] = (None, None)
	defValues["acf.diff"] = (False, None)
	defValues["pacf.lags"] = (40, None)
	defValues["pacf.alpha"] = (None, None)
	defValues["ccf.normed"] = (True, None)
	defValues["ccf.maxlags"] = (10, None)
	defValues["adf.regression"] = ("c", None)
	defValues["adf.autolag"] = ("AIC", None)
	defValues["kpss.regression"] = ("c", None)
	defValues["hist.cumulative"] = (False, None)
	defValues["hist.density"] = (False, None)
	defValues["cov.file.paths"] = (None, "missing list of file path and column index")
	
	config = Configuration(configFile, defValues)
	return config

def appendKey(key, extra):
	"""
	appends config key
	"""
	if extra:
		key = key + ".extra"
	return key	

def loadData(config, extra=False):
	"""
	loads a column
	"""
	key = "data.filePath"
	key = appendKey(key, extra)
	filePath = config.getStringConfig(key)[0]
	
	key = "data.col.index"
	key = appendKey(key, extra)
	col = config.getIntConfig(key)[0]
	data = getFileColumnAsFloat(filePath ,col,  ",")
	
	
	key = "data.row.range"
	key = appendKey(key, extra)
	if not config.getStringConfig(key)[0] == "all":
		ra = config.getIntListConfig(key, ",")[0]
		data = data[ra[0]:ra[1]]
	return np.array(data)

def getConTab(config):
	"""
	get contingency table
	"""
	filePath = config.getStringConfig("data.filePath")[0]
	otherFilePath = config.getStringConfig("data.filePath.extra")[0]
	c1 = config.getIntConfig("data.col.index")[0]
	c2 = config.getIntConfig("data.col.index.extra")[0]
	if filePath == otherFilePath:
		data = pd.read_csv(filePath,  header=None) 
		data = data.loc[ : , [c1, c2]]
		data.columns = range(data.shape[1])
	else:
		d1 = pd.read_csv(filePath, header=None) 
		d2 = pd.read_csv(otherFilePath, header=None) 
		d1 = d1.loc[ : , c1 ]
		d2 = d2.loc[ : , c2 ]
		data = pd.concat([d1,d2], axis=1)
		data.columns = range(data.shape[1])

	crosstab = pd.crosstab(data[0], data[1], margins = False)
	ctab = crosstab.values
	return ctab

def printStat(stat, pvalue, nhMsg, ahMsg):
	"""
	generic stat and pvalue output
	"""
	print("stat:   {:.3f}".format(stat))
	print("pvalue: {:.3f}".format(pvalue))
	print(nhMsg if pvalue > 0.05 else ahMsg)

if __name__ == "__main__":
	op = sys.argv[1]
	confFile = sys.argv[2]
	config = loadConfig(confFile)
	noLoadOps = set(["cov", "desc", "contab", "cscorr"])
	if op not in noLoadOps:
		data = loadData(config)

	if op == "desc":
		filePath = config.getStringConfig("data.filePath")[0]
		data = pd.read_csv(filePath, header=None) 
		print (data.head(5)) 
		print(data.describe())
		print(data.info())

	#plot data
	elif op == "draw":
		pyplot.plot(data)
		pyplot.show()
	
	#difference
	elif op == "diff":
		order = config.getIntConfig("diff.order")[0]
		for i in range(order):
			data = difference(data)
		pyplot.plot(data)
		pyplot.show()

	#trend with linear regression
	elif op == "trend":
		sz = len(data)
		X = list(range(0, sz))
		X = np.reshape(X, (sz, 1))
		model = LinearRegression()
		model.fit(X, data)
		trend = model.predict(X)
		sc = model.score(X, data)
		coef = model.coef_
		intc = model.intercept_
		print("R square {:.6f}".format(sc))
		print("intercept  {:.6f} coeffficient {:.6f} ".format(intc, coef[0]))
		pyplot.plot(data)
		pyplot.plot(trend)
		pyplot.show()
		
		remTrend = config.getBooleanConfig("trend.remove")[0]
		if remTrend:
			detrended = [data[i]-trend[i] for i in range(0, sz)]
			pyplot.plot(detrended)
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

	#cross correlation
	elif op == "ccf":
		dataSec = loadData(config, True)
		normed = config.getBooleanConfig("ccf.normed")[0]
		maxlags = config.getIntConfig("ccf.maxlags")[0]
		pyplot.xcorr(data, dataSec, normed=normed, maxlags=maxlags)
		pyplot.show()
		
		
	#stationarity test
	elif op == "adf":
		regression = config.getStringConfig("adf.regression")[0]
		autolag = config.getStringConfig("adf.autolag")[0]
		result = adfuller(data, regression=regression, autolag=autolag)
		printStat(result[0], result[1], "probably not stationary", "probably stationary")
		print(f'num lags: {result[2]}')
		print(f'num observation for regression: {result[3]}')
		print('critial values:')
		for key, value in result[4].items():
			print(f'   {key}, {value}')    
			
	#stationarity test
	elif op =="kpss":
		regression = config.getStringConfig("kpss.regression")[0]
		stat, pvalue, nLags, criticalValues = kpss(data, regression=regression)
		printStat(stat, pvalue, "probably not stationary", "probably stationary")
		print(f'num lags: {nLags}')
		print('critial values:')
		for key, value in criticalValues.items():
			print(f'   {key} : {value}')	
				
	#normalcy test	
	elif op == "jarqBera":	
		jb, jbpv, skew, kurtosis =  jarque_bera(data)
		printStat(jb, jbpv, "probably gaussian", "probably not gaussian")
		print(f'skew: {skew}')
		print(f'kurtosis: {kurtosis}')
		
	#normalcy test
	elif op == "shapWilk":	
		stat, pvalue = shapiro(data)
		printStat(stat, pvalue, "probably gaussian", "probably not gaussian")

	#histogram
	elif op == "hist":	
		cumulative = config.getBooleanConfig("hist.cumulative")[0]
		density = config.getBooleanConfig("hist.density")[0]
		pyplot.hist(data, cumulative=cumulative, density=density)
		pyplot.show()

	#co variance
	elif op == "cov":
		pcListStr = config.getStringConfig("cov.file.paths")[0].split(",")
		pathColList = list()
		for pc in pcListStr:
			items = pc.split(":")
			pathCol = (items[0], int(items[1]))
			pathColList.append(pathCol)
		values = asNumpyArray(getMultipleFileAsFloatMatrix(pathColList))
		print(values)
		cov = np.cov(values)
		print("co variance matrix")
		print(cov)

	#pearson correlation	
	elif op == "pcorr":
		dataSec = loadData(config, True)
		stat, pvalue = pearsonr(data, dataSec)
		printStat(stat, pvalue, "probably uncorrelated", "probably correlated")
	
	#spearman rank correlation	
	elif op == "srcorr":
		dataSec = loadData(config, True)
		stat, pvalue = spearmanr(data, dataSec)
		printStat(stat, pvalue, "probably uncorrelated", "probably correlated")

	#kendal rank correlation	
	elif op == "krcorr":
		dataSec = loadData(config, True)
		stat, pvalue = kendalltau(data, dataSec)
		printStat(stat, pvalue, "probably uncorrelated", "probably correlated")
	
	#chi square correlation for categorical	
	elif op == "cscorr":
		ctab = getConTab(config)
		stat, pvalue, dof, expctd = chi2_contingency(ctab)
		printStat(stat, pvalue, "probably uncorrelated", "probably correlated")
		print("dof " + str(dof))
		print("actual")
		print(ctab)
		print("expected")
		print(expctd)

	#contingency table	
	elif op == "contab":
		ctab = getConTab(config)
		print(ctab)

	#Kolmogorov Sminov 2 sample statistic	
	elif op == "ks2s":
		dataSec = loadData(config, True)
		stat, pvalue = ks_2samp(data, dataSec)
		printStat(stat, pvalue, "probably same distribution", "probably same distribution")
		
	#Mann-Whitney U 2 sample statistic	
	elif op == "mawh":
		dataSec = loadData(config, True)
		stat, pvalue = mannwhitneyu(data, dataSec)
		printStat(stat, pvalue, "probably same distribution", "probably same distribution")

	#Wilcoxon Signed-Rank 2 sample statistic	
	elif op == "wilcox":
		dataSec = loadData(config, True)
		stat, pvalue = wilcoxon(data, dataSec)
		printStat(stat, pvalue, "probably same distribution", "probably same distribution")

	else:
		raise ValueError("unknown command")
		
		