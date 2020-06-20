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
		list data
		"""
		if type(dataSet) == list:
			dataSet = np.array(dataSet)
		self.dataSets[name] = dataSet


	def addFileCatData(self, filePath,  *columns):
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
				col = df.loc[ : , ci].tolist()
				cn = columns[i + nCols]
				#print(ci,cn)
				self.dataSets[cn] = col
		else:
			df = pd.read_csv(filePath,  header=0) 
			for c in columns:
				col = df[c].tolist()
				self.dataSets[c] = col

	def addDataFrameCatData(self, df,  *columns):
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
				col = df.loc[ : , ci].tolist()
				cn = columns[i + nCols]
				self.dataSets[cn] = col
		else:
			for c in columns:
				col = df[c].tolist()
				self.dataSets[c] = col

	def addCatListData(self, ds, name):
		"""
		categorical list data
		"""
		self.dataSets[name] = ds

	def getData(self, ds):
		"""
		get data
		"""
		if type(ds) == str:
			assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
			data =   self.dataSets[ds]
		elif type(ds) == list:
			data = np.array(ds)
		elif type(ds) == numpy.ndarray:
			data = ds
		else:
			raise "invalid type, expecting data set name, list or ndarray"			
		return data


	def getCatData(self, ds):
		"""
		get data
		"""
		if type(ds) == str:
			assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
			data =   self.dataSets[ds]
		elif type(ds) == list:
			data = ds
		else:
			raise "invalid type, expecting data set name or list"
		return data

	def loadCatFloatDataFrame(self, ds1, ds2):
		"""
		loads float and cat data into data frame
		"""
		data1 = self.getCatData(ds1)
		data2 = self.getData(ds2)
		df1 = pd.DataFrame(data=data1)
		df2 = pd.DataFrame(data=data2)
		df = pd.concat([df1,df2], axis=1)
		df.columns = range(df.shape[1])
		return df

	def plot(self, ds, yscale=None):
		"""
		plots data
		"""
		data = self.getData(ds)
		drawLine(data, yscale)

	def plotHist(self, ds, cumulative, density, nbins=None):
		"""
		plots data
		"""
		data = self.getData(ds)
		plt.hist(data, bins=nbins, cumulative=cumulative, density=density)
		plt.show()

	def getStats(self, ds):
		"""
		plots data
		"""
		data = self.getData(ds)
		stat = dict()
		stat["length"] = len(data)
		stat["min"] = data.min()
		stat["max"] = data.max()
		stat["mean"] = data.mean()
		stat["median"] = np.median(data)
		stat["mean"] = data.mean()
		self.pp.pprint(stat)
		return stat


	def getDifference(self, ds, order):
		"""
		difference of given order
		"""
		data = self.getData(ds)
		diff = difference(data, order)
		drawLine(diff)
		return diff

	def getTrend(self, ds, doPlot=False):
		"""
		finds trend
		"""
		data = self.getData(ds)
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

	def deTrend(self, ds, trend, doPlot=False):
		"""
		de trend
		"""
		data = self.getData(ds)
		sz = len(data)
		detrended =  list(map(lambda i : data[i]-trend[i], range(sz)))
		if doPlot:
			drawLine(detrended)
		return detrended

	def getCovar(self, *dsl):
		"""
		covariance
		"""
		data = list(map(lambda ds : self.getData(ds), dsl))
		data = np.vstack(data)
		cv = np.cov(data)
		print(cv)
		return cv

	def getPearsonCorr(self, ds1, ds2, sigLev=.05):
		"""
		covariance
		"""
		data1 = self.getData(ds1)
		data2 = self.getData(ds2)
		stat, pvalue = sta.pearsonr(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result


	def getSpearmanRankCorr(self, ds1, ds2, sigLev=.05):
		"""
		covariance
		"""
		data1 = self.getData(ds1)
		data2 = self.getData(ds2)
		stat, pvalue = sta.spearmanr(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getKendalRankCorr(self, ds1, ds2, sigLev=.05):
		"""
		covariance
		"""
		data1 = self.getData(ds1)
		data2 = self.getData(ds2)
		stat, pvalue = sta.kendalltau(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getConTab(self, ds1, ds2):
		"""
		get contingency table
		"""
		data1 = getCatData(ds1)
		data2 = getCatData(ds2)
		crosstab = pd.crosstab(data1, data2, margins = False)
		ctab = crosstab.values
		print("contingency table")
		print(ctab)
		return ctab

	def getChiSqCorr(self, ds1, ds2, sigLev=.05):
		"""
		chi square correlation for  both categorical	
		"""
		ctab = self.getConTab(ds1, ds2)
		stat, pvalue, dof, expctd = sta.chi2_contingency(ctab)
		result = self.printResult("stat", stat, "pvalue", pvalue, "dof", dof, "expected", expctd)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getAnovaCorr(self, ds1, ds2, grByCol, sigLev=.05):
		"""
		anova correlation for  numerical categorical	
		"""
		df = loadCatFloatDataFrame(ds1, ds2) if grByCol == 0 else loadCatFloatDataFrame(ds2, ds1)
		grByCol = 0
		dCol = 1
		grouped = df.groupby([grByCol])
		dlist =  list(map(lambda v : v[1].loc[:, dCol].values, grouped))
		stat, pvalue = f_oneway(*dlist)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result


	def plotAcf(self, ds, lags, alpha, diffOrder=0):
		"""
		auto correlation
		"""
		data = self.getData(ds)
		ddata = difference(data, diffOrder) if diffOrder > 0 else data
		tsaplots.plot_acf(ddata, lags = lags, alpha = alpha)
		plt.show()

	def plotParAcf(self, ds, lags, alpha):
		"""
		partial auto correlation
		"""
		data = self.getData(ds)
		tsaplots.plot_pacf(data, lags = lags, alpha = alpha)
		plt.show()

	def plotCrossCorr(self, dsOne, dsTwo, normed, lags):
		"""
		cross correlation 
		"""
		dataOne = self.getData(dsOne)
		dataTwo = self.getData(dsTwo)
		plt.xcorr(dataOne, dataTwo, normed=normed, maxlags=lags)
		plt.show()

	def testStationaryAdf(self, ds, regression, autolag, sigLev=.05):
		"""
		Adf stationary test null hyp not stationary
		"""
		data = self.getData(ds)
		re = adfuller(data, regression=regression, autolag=autolag)
		result = self.printResult("stat", re[0], "pvalue", re[1] , "num lags", re[2] , "num observation for regression", re[3],
		"critial values", re[4])
		self.printStat(re[0], re[1], "probably not stationary", "probably stationary", sigLev)
		return result

	def testStationaryKpss(self, ds, regression, sigLev=.05):
		"""
		Kpss stationary test null hyp  stationary
		"""
		data = self.getData(ds)
		stat, pvalue, nLags, criticalValues = kpss(data, regression=regression)
		result = self.printResult("stat", stat, "pvalue", pvalue, "num lags", nLags, "critial values", criticalValues)
		self.printStat(stat, pvalue, "probably stationary", "probably not stationary", sigLev)
		return result

	def testNormalJarqBera(self, ds, sigLev=.05):
		"""
		jarque bera normalcy test
		"""
		data = self.getData(ds)
		jb, jbpv, skew, kurtosis =  jarque_bera(data)
		result = self.printResult("stat", jb, "pvalue", jbpv, "skew", skew, "kurtosis", kurtosis)
		self.printStat(jb, jbpv, "probably gaussian", "probably not gaussian", sigLev)
		return result


	def testNormalShapWilk(self, ds, sigLev=.05):
		"""
		shapiro wilks normalcy test
		"""
		data = self.getData(ds)
		stat, pvalue = shapiro(data)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably gaussian", "probably not gaussian", sigLev)
		return result

	def testNormalDagast(self, ds, sigLev=.05):
		"""
		D’Agostino’s K square  normalcy test
		"""
		data = self.getData(ds)
		stat, pvalue = normaltest(data)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably gaussian", "probably not gaussian", sigLev)
		return result

	def testNormalAnderson(self, ds, sigLev=.05):
		"""
		D’Agostino’s K square  normalcy test
		"""
		data = self.getData(ds)
		re = anderson(data)
		slAlpha = int(100 * sigLev)
		msg = "significnt value not found"
		for i in range(len(re.critical_values)):
			sl, cv = re.significance_level[i], re.critical_values[i]
			if int(sl) == slAlpha:
				if re.statistic < cv:
					msg = "probably gaussian at the {:.1f} level".format(sl)
				else:
					msg = "probably not gaussian at the {:.1f} level".format(sl)
		result = self.printResult("stat", re.statistic, "test", msg)
		print(msg)
		return result


	def testTwoSampleCvm(self, ds1, ds2, sigLev=.05):
		"""
		2 sample cramer von mises
		"""
		pass
		

	def printStat(self, stat, pvalue, nhMsg, ahMsg, sigLev=.05):
		"""
		generic stat and pvalue output
		"""
		print("test result:")
		print("stat:   {:.3f}".format(stat))
		print("pvalue: {:.3f}".format(pvalue))
		print(nhMsg if pvalue > sigLev else ahMsg)

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


