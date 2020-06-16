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
 
# Package imports
import os
import sys
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
from sklearn import metrics
import random
from math import *
from decimal import Decimal
#import jprops
import pprint
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.stattools import jarque_bera
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from scipy import stats as sta
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import shapiro
from scipy.stats import kendalltau
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy.stats import friedmanchisquare
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import normaltest
from scipy.stats import anderson
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *

class DataExplorer:
	"""
	various data exploration functions
	"""
	def __init__(self):
		self.dataSets = dict()
		self.pp = pprint.PrettyPrinter(indent=4)


	def addFileData(self,filePath,  *columns):
		"""
		add columns from a file
		"""
		columns = list(columns)
		noHeader = type(columns[0]) ==  int
		if noHeader:			
			nCols = int(len(columns) / 2)
			colIndexes = columns[:nCols]
			df = pd.read_csv(filePath,  header=None) 
			nColsDf = len(df.columns)
			for i in range(nCols):
				ci = colIndexes[i]
				assert ci < nColsDf, "col index {} outside range".format(ci)
				col = df.loc[ : , ci].to_numpy()
				cn = columns[i + nCols]
				#print(ci,cn)
				self.dataSets[cn] = col
		else:
			df = pd.read_csv(filePath,  header=0) 
			for c in columns:
				col = df[c].to_numpy()
				self.dataSets[c] = col



	def addDataFrameData(self, df,  *columns):
		"""
		add columns from a data frame
		"""
		columns = list(columns)
		noHeader = type(columns[0]) ==  int
		if noHeader:			
			nCols = int(len(columns) / 2)
			colIndexes = columns[:nCols]
			nColsDf = len(df.columns)
			for i in range(nCols):
				ci = colIndexes[i]
				assert ci < nColsDf, "col index {} outside range".format(ci)
				col = df.loc[ : , ci].to_numpy()
				cn = columns[i + nCols]
				self.dataSets[cn] = col
		else:
			for c in columns:
				col = df[c].to_numpy()
				self.dataSets[c] = col

	def addListData(self, dataSet, name):
		"""
		add columns for a file
		"""
		if type(dataSet) == list:
			dataSet = np.array(dataSet)
		self.dataSets[name] = dataSet

	def getData(self, dname):
		"""
		get data
		"""
		assert dname in self.dataSets, "data set {} does not exist".format(dname)
		return  self.dataSets[dname]
			
	def plot(self, dname, yscale=None):
		"""
		plots data
		"""
		data = self.getData(dname)
		drawLine(data, yscale)

	def getStats(self, dname):
		"""
		plots data
		"""
		data = self.getData(dname)
		stat = dict()
		stat["length"] = len(data)
		stat["min"] = data.min()
		stat["max"] = data.max()
		stat["mean"] = data.mean()
		stat["median"] = np.median(data)
		stat["mean"] = data.mean()
		self.pp.pprint(stat)
		return stat


	def getDifference(self, dname, order):
		"""
		difference of given order
		"""
		data = data = self.getData(dname)
		diff = difference(data, order)
		drawLine(diff)
		return diff

	def getTrend(self, dname, doPlot=False):
		"""
		finds trend
		"""
		data = self.getData(dname)
		sz = len(data)
		X = list(range(0, sz))
		X = np.reshape(X, (sz, 1))
		model = LinearRegression()
		model.fit(X, data)
		trend = model.predict(X)
		sc = model.score(X, data)
		coef = model.coef_
		intc = model.intercept_

		result = dict()
		result["coeff"] = coef
		result["intercept"] = intc
		result["r square error"] = sc
		result["trend"] = trend
		self.pp.pprint(result)
		
		if doPlot:
			plt.plot(data)
			plt.plot(trend)
			plt.show()
		return result

	def deTrend(self, dname, trend, doPlot=False):
		"""
		de trend
		"""
		data = self.getData(dname)
		sz = len(data)
		detrended =  list(map(lambda i : data[i]-trend[i], range(sz)))
		if doPlot:
			drawLine(detrended)
		return detrended

	def plotAcf(self, dname, lags, alpha, diffOrder=0):
		"""
		auto correlation
		"""
		data = self.getData(dname)
		ddata = difference(data, diffOrder) if diffOrder > 0 else data
		tsaplots.plot_acf(ddata, lags = lags, alpha = alpha)
		plt.show()

	def plotParAcf(self, dname, lags, alpha):
		"""
		partial auto correlation
		"""
		data = self.getData(dname)
		tsaplots.plot_pacf(data, lags = lags, alpha = alpha)
		plt.show()

	def plotCrossCorr(self, dnameOne, dnameTwo, normed, lags):
		"""
		cross correlation 
		"""
		dataOne = self.getData(dnameOne)
		dataTwo = self.getData(dnameTwo)
		plt.xcorr(dataOne, dataTwo, normed=normed, maxlags=lags)
		plt.show()

	def testStationaryAdf(self, dname, regression, autolag, critValue=.05):
		"""
		Adf stationary test null hyp not stationary
		"""
		data = self.getData(dname)
		re = adfuller(data, regression=regression, autolag=autolag)
		result = self.printResult("stat", re[0], "pvalue", re[1] , "num lags", re[2] , "num observation for regression", re[3],
		"critial values", re[4])
		self.printStat(re[0], re[1], "probably not stationary", "probably stationary", critValue)
		return result

	def testStationaryKpss(self, dname, regression, critValue=.05):
		"""
		Kpss stationary test null hyp  stationary
		"""
		data = self.getData(dname)
		stat, pvalue, nLags, criticalValues = kpss(data, regression=regression)
		result = self.printResult("stat", stat, "pvalue", pvalue, "num lags", nLags, "critial values", criticalValues)
		self.printStat(stat, pvalue, "probably stationary", "probably not stationary", critValue)
		return result

	def testNormalJarqBera(self, dname, critValue=.05):
		"""
		jarque bera normalcy test
		"""
		data = self.getData(dname)
		jb, jbpv, skew, kurtosis =  jarque_bera(data)
		result = self.printResult("stat", jb, "pvalue", jbpv, "skew", skew, "kurtosis", kurtosis)
		self.printStat(jb, jbpv, "probably gaussian", "probably not gaussian", critValue)
		return result

	def printStat(self, stat, pvalue, nhMsg, ahMsg, critVal=.05):
		"""
		generic stat and pvalue output
		"""
		print("test result:")
		print("stat:   {:.3f}".format(stat))
		print("pvalue: {:.3f}".format(pvalue))
		print(nhMsg if pvalue > critVal else ahMsg)

	def printResult(self,  *values):
		"""
		print results
		"""
		result = dict()
		for i in range(0, len(values), 2):
			result[values[i]] = values[i+1]
		print("result details:")
		self.pp.pprint(result)
		return result


