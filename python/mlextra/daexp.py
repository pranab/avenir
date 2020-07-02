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
import pprint
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.stattools import jarque_bera
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from scipy import stats as sta
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

	def save(self, filePath):
		"""
		save checkpoint
		"""
		saveObject(self.dataSets, filePath)

	def restore(self, filePath):
		"""
		restore checkpoint
		"""
		self.dataSets = restoreObject(filePath)


	def queryFileData(self, filePath,  *columns):
		"""
		query column data type  from a data frame
		"""
		lcolumns = list(columns)
		noHeader = type(lcolumns[0]) ==  int
		if noHeader:			
			df = pd.read_csv(filePath,  header=None) 
		else:
			df = pd.read_csv(filePath,  header=0) 
		return self.queryDataFrameData(df,  *columns)

	def queryDataFrameData(self, df,  *columns):
		"""
		query column data type  from a data frame
		"""
		columns = list(columns)
		noHeader = type(columns[0]) ==  int
		dtypes = list()
		if noHeader:			
			nCols = int(len(columns) / 2)
			colIndexes = columns[:nCols]
			cnames = columns[nCols:]
			nColsDf = len(df.columns)
			for i in range(nCols):
				ci = colIndexes[i]
				assert ci < nColsDf, "col index {} outside range".format(ci)
				col = df.loc[ : , ci]
				dtypes.append(self.getDataType(col))
		else:
			cnames = columns
			for c in columns:
				col = df[c]
				dtypes.append(self.getDataType(col))

		nt = list(zip(cnames, dtypes))
		result = self.printResult("columns and data types", nt)
		return result

	def getDataType(self, col):
		"""
		get data type 
		"""
		if isBinary(col):
			dtype = "binary"
		elif  isInteger(col):
			dtype = "integer"
		elif  isFloat(col):
			dtype = "float"
		elif  isCategorical(col):
			dtype = "categorical"
		else:
			dtype = "mixed"
		return dtype


	def addFileNumericData(self,filePath,  *columns):
		"""
		add numeric columns from a file
		"""
		self.addFileData(filePath, True, *columns)


	def addFileBinaryData(self,filePath,  *columns):
		"""
		add binary columns from a file
		"""
		self.addFileData(filePath, False, *columns)

	def addFileData(self, filePath,  numeric, *columns):
		"""
		add columns from a file
		"""
		columns = list(columns)
		noHeader = type(columns[0]) ==  int
		if noHeader:			
			df = pd.read_csv(filePath,  header=None) 
		else:
			df = pd.read_csv(filePath,  header=0) 
		self.addDataFrameData(df, numeric, *columns)

	def addDataFrameNumericData(self,filePath,  *columns):
		"""
		add numeric columns from a file
		"""
		self.addDataFrameData(filePath, True, *columns)


	def addDataFrameBinaryData(self,filePath,  *columns):
		"""
		add binary columns from a file
		"""
		self.addDataFrameData(filePath, False, *columns)


	def addDataFrameData(self, df,  numeric, *columns):
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
				col = df.loc[ : , ci]
				if numeric:
					assert isNumeric(col), "data is not numeric"
				else:
					assert isBinary(col), "data is not binary"
				col = col.to_numpy()
				cn = columns[i + nCols]
				self.dataSets[cn] = col
		else:
			for c in columns:
				col = df[c]
				if numeric:
					assert isNumeric(col), "data is not numeric"
				else:
					assert isBinary(col), "data is not binary"
				col = col.to_numpy()
				self.dataSets[c] = col

	def addListNumericData(self, ds,  name):
		"""
		add numeric columns from a file
		"""
		self.addListData(ds, True,  name)


	def addListBinaryData(self, ds, name):
		"""
		add binary columns from a file
		"""
		self.addListData(ds, False,  name)

	def addListData(self, ds, numeric,  name):
		"""
		list data
		"""
		assert type(ds) == list, "data not a list"
		if numeric:
			assert isNumeric(ds), "data is not numeric"
		else:
			assert isBinary(ds), "data is not binary"
		self.dataSets[name] = np.array(ds)


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
				col = df.loc[ : , ci]
				assert isCategorical(col), "data is not categorical"
				col = col.tolist()
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
				col = df.loc[ : , ci]
				assert isCategorical(col), "data is not categorical"
				col = col.tolist()
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
		assert type(ds) == list, "data not a list"
		assert isCategorical(ds), "data is not categorical"
		self.dataSets[name] = ds

	def remData(self, ds):
		"""
		removes data set
		"""
		assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
		self.dataSets.pop(ds)
		self.showNames()

	def getNumericData(self, ds):
		"""
		get data
		"""
		if type(ds) == str:
			assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
			data =   self.dataSets[ds]
		elif type(ds) == list:
			assert isNumeric(ds), "data is not numeric"
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
			assert isCategorical(ds), "data is not categorical"
			data = ds
		else:
			raise "invalid type, expecting data set name or list"
		return data

	def loadCatFloatDataFrame(self, ds1, ds2):
		"""
		loads float and cat data into data frame
		"""
		data1 = self.getCatData(ds1)
		data2 = self.getNumericData(ds2)
		df1 = pd.DataFrame(data=data1)
		df2 = pd.DataFrame(data=data2)
		df = pd.concat([df1,df2], axis=1)
		df.columns = range(df.shape[1])
		return df

	def showNames(self):
		"""
		lists data set names
		"""
		print("data sets")
		for ds in self.dataSets.keys():
			print(ds)

	def plot(self, ds, yscale=None):
		"""
		plots data
		"""
		data = self.getNumericData(ds)
		drawLine(data, yscale)

	def print(self, ds):
		"""
		prunt data
		"""
		assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
		data =   self.dataSets[ds]
		print(formatAny(len(data), "size"))
		print("showing first 50 elements" )
		print(data[:50])

	def plotHist(self, ds, cumulative, density, nbins=None):
		"""
		plots data
		"""
		data = self.getNumericData(ds)
		plt.hist(data, bins=nbins, cumulative=cumulative, density=density)
		plt.show()

	def getPercentile(self, ds, value):
		"""
		gets percentile
		"""
		data = self.getNumericData(ds)
		percent = sta.percentileofscore(data, value)
		result = self.printResult("value", value, "percentile", percent)
		return result

	def getValueAtPercentile(self, ds, percent):
		"""
		gets value at percentile
		"""
		data = self.getNumericData(ds)
		assert isInRange(percent, 0, 100), "percent should be between 0 and 100"
		value = sta.scoreatpercentile(data, percent)
		result = self.printResult("value", value, "percentile", percent)
		return result

	def getUniqueValueCounts(self, ds, maxCnt=10):
		"""
		gets unique values and counts
		"""
		data = self.getNumericData(ds)
		values, counts = sta.find_repeats(data)
		cardinality = len(values)
		vc = list(zip(values, counts))
		vc.sort(key = lambda v : v[1], reverse = True)
		result = self.printResult("cardinality", cardinality,  "vunique alues and repeat counts", vc[:maxCnt])
		return result

	def getCatUniqueValueCounts(self, ds, maxCnt=10):
		"""
		gets unique categorical values and counts
		"""
		data = self.getCatData(ds)
		series = pd.Series(data)
		uvalues = series.value_counts()
		values = uvalues.index.tolist()
		counts = uvalues.tolist()
		vc = list(zip(values, counts))
		vc.sort(key = lambda v : v[1], reverse = True)
		result = self.printResult("cardinality", len(values),  "unique values and repeat counts", vc[:maxCnt])
		return result

	def getStats(self, ds, nextreme=5):
		"""
		plots data
		"""
		data = self.getNumericData(ds)
		stat = dict()
		stat["length"] = len(data)
		stat["min"] = data.min()
		stat["max"] = data.max()
		series = pd.Series(data)
		stat["n smallest"] = series.nsmallest(n=nextreme).tolist()
		stat["n largest"] = series.nlargest(n=nextreme).tolist()
		stat["mean"] = data.mean()
		stat["median"] = np.median(data)
		mode, modeCnt = sta.mode(data)
		stat["mode"] = mode[0]
		stat["mode count"] = modeCnt[0]
		stat["std"] = np.std(data)
		stat["skew"] = sta.skew(data)
		stat["kurtosis"] = sta.kurtosis(data)
		stat["mad"] = sta.median_absolute_deviation(data)
		self.pp.pprint(stat)
		return stat


	def getDifference(self, ds, order):
		"""
		difference of given order
		"""
		data = self.getNumericData(ds)
		diff = difference(data, order)
		drawLine(diff)
		return diff

	def getTrend(self, ds, doPlot=False):
		"""
		finds trend
		"""
		data = self.getNumericData(ds)
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
		data = self.getNumericData(ds)
		sz = len(data)
		detrended =  list(map(lambda i : data[i]-trend[i], range(sz)))
		if doPlot:
			drawLine(detrended)
		return detrended

	def fitLinearReg(self, ds, doPlot=False):
		"""
		fit  linear regression 
		"""
		data = self.getNumericData(ds)
		x = np.arange(len(data))
		slope, intercept, rvalue, pvalue, stderr = sta.linregress(x, data)
		result = self.printResult("slope", slope, "intercept", intercept, "rvalue", rvalue, "pvalue", pvalue, "stderr", stderr)
		if doPlot:
			self.regFitPlot(x, data, slope, intercept)
		return result

	def fitSiegelRobustLinearReg(self, ds, doPlot=False):
		"""
		siegel robust linear regression fit based on median
		"""
		data = self.getNumericData(ds)
		slope , intercept = sta.siegelslopes(data)
		result = self.printResult("slope", slope, "intercept", intercept)
		if doPlot:
			x = np.arange(len(data))
			self.regFitPlot(x, data, slope, intercept)
		return result

	def fitTheilSenRobustLinearReg(self, ds, doPlot=False):
		"""
		thiel sen  robust linear fit regression based on median
		"""
		data = self.getNumericData(ds)
		slope, intercept, loSlope, upSlope = sta.theilslopes(data)
		result = self.printResult("slope", slope, "intercept", intercept, "lower slope", loSlope, "upper slope", upSlope)
		if doPlot:
			x = np.arange(len(data))
			self.regFitPlot(x, data, slope, intercept)
		return result

	def regFitPlot(self, x, y, slope, intercept):
		"""
		plot linear rgeression fit line
		"""
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(x, y, "b.")
		ax.plot(x, intercept + slope * x, "r-")
		plt.show()

	def getCovar(self, *dsl):
		"""
		covariance
		"""
		data = list(map(lambda ds : self.getNumericData(ds), dsl))
		data = np.vstack(data)
		cv = np.cov(data)
		print(cv)
		return cv

	def getPearsonCorr(self, ds1, ds2, sigLev=.05):
		"""
		pearson correlation coefficient 
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.pearsonr(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result


	def getSpearmanRankCorr(self, ds1, ds2, sigLev=.05):
		"""
		spearman correlation coefficient
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.spearmanr(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getKendalRankCorr(self, ds1, ds2, sigLev=.05):
		"""
		kendall’s tau, a correlation measure for ordinal data
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.kendalltau(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getPointBiserialCorr(self, ds1, ds2, sigLev=.05):
		"""
		kendall’s tau, a correlation measure for ordinal data
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		assert isBinary(data1), "first data set is not binary"
		stat, pvalue = sta.pointbiserialr(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getConTab(self, ds1, ds2):
		"""
		get contingency table
		"""
		data1 = self.getCatData(ds1)
		data2 = self.getCatData(ds2)
		crosstab = pd.crosstab(pd.Series(data1), pd.Series(data2), margins = False)
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
		df = self.loadCatFloatDataFrame(ds1, ds2) if grByCol == 0 else self.loadCatFloatDataFrame(ds2, ds1)
		grByCol = 0
		dCol = 1
		grouped = df.groupby([grByCol])
		dlist =  list(map(lambda v : v[1].loc[:, dCol].values, grouped))
		stat, pvalue = sta.f_oneway(*dlist)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result


	def plotAcf(self, ds, lags, alpha, diffOrder=0):
		"""
		auto correlation
		"""
		data = self.getNumericData(ds)
		ddata = difference(data, diffOrder) if diffOrder > 0 else data
		tsaplots.plot_acf(ddata, lags = lags, alpha = alpha)
		plt.show()

	def plotParAcf(self, ds, lags, alpha):
		"""
		partial auto correlation
		"""
		data = self.getNumericData(ds)
		tsaplots.plot_pacf(data, lags = lags, alpha = alpha)
		plt.show()

	def plotCrossCorr(self, dsOne, dsTwo, normed, lags):
		"""
		cross correlation 
		"""
		dataOne = self.getNumericData(dsOne)
		dataTwo = self.getNumericData(dsTwo)
		plt.xcorr(dataOne, dataTwo, normed=normed, maxlags=lags)
		plt.show()

	def testStationaryAdf(self, ds, regression, autolag, sigLev=.05):
		"""
		Adf stationary test null hyp not stationary
		"""
		data = self.getNumericData(ds)
		re = adfuller(data, regression=regression, autolag=autolag)
		result = self.printResult("stat", re[0], "pvalue", re[1] , "num lags", re[2] , "num observation for regression", re[3],
		"critial values", re[4])
		self.printStat(re[0], re[1], "probably not stationary", "probably stationary", sigLev)
		return result

	def testStationaryKpss(self, ds, regression, sigLev=.05):
		"""
		Kpss stationary test null hyp  stationary
		"""
		data = self.getNumericData(ds)
		stat, pvalue, nLags, criticalValues = kpss(data, regression=regression)
		result = self.printResult("stat", stat, "pvalue", pvalue, "num lags", nLags, "critial values", criticalValues)
		self.printStat(stat, pvalue, "probably stationary", "probably not stationary", sigLev)
		return result

	def testNormalJarqBera(self, ds, sigLev=.05):
		"""
		jarque bera normalcy test
		"""
		data = self.getNumericData(ds)
		jb, jbpv, skew, kurtosis =  jarque_bera(data)
		result = self.printResult("stat", jb, "pvalue", jbpv, "skew", skew, "kurtosis", kurtosis)
		self.printStat(jb, jbpv, "probably gaussian", "probably not gaussian", sigLev)
		return result


	def testNormalShapWilk(self, ds, sigLev=.05):
		"""
		shapiro wilks normalcy test
		"""
		data = self.getNumericData(ds)
		stat, pvalue = sta.shapiro(data)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably gaussian", "probably not gaussian", sigLev)
		return result

	def testNormalDagast(self, ds, sigLev=.05):
		"""
		D’Agostino’s K square  normalcy test
		"""
		data = self.getNumericData(ds)
		stat, pvalue = sta.normaltest(data)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably gaussian", "probably not gaussian", sigLev)
		return result

	def testDistrAnderson(self, ds, dist, sigLev=.05):
		"""
		Anderson   test for normal, expon, logistic, gumbel, gumbel_l, gumbel_r
		"""
		data = self.getNumericData(ds)
		re = sta.anderson(data)
		slAlpha = int(100 * sigLev)
		msg = "significnt value not found"
		for i in range(len(re.critical_values)):
			sl, cv = re.significance_level[i], re.critical_values[i]
			if int(sl) == slAlpha:
				if re.statistic < cv:
					msg = "probably gaussian at the {:.3f} siginificance level".format(sl)
				else:
					msg = "probably not gaussian at the {:.3f} siginificance level".format(sl)
		result = self.printResult("stat", re.statistic, "test", msg)
		print(msg)
		return result

	def testSkew(self, ds, sigLev=.05):
		"""
		test skew wrt  normal distr
		"""
		data = self.getNumericData(ds)
		stat, pvalue = sta.skewtest(data)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same skew as normal distribution", "probably not same skew as normal distribution", sigLev)
		return result

	def testTwoSampleStudent(self, ds1, ds2, sigLev=.05):
		"""
		student t 2 sample test
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.ttest_ind(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)


	def testTwoSampleKs(self, ds1, ds2, sigLev=.05):
		"""
		Kolmogorov Sminov 2 sample statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.ks_2samp(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)


	def testTwoSampleMw(self, ds1, ds2, sigLev=.05):
		"""
		Mann-Whitney  2 sample statistic
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.mannwhitneyu(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)

	def testTwoSampleWilcox(self, ds1, ds2, sigLev=.05):
		"""
		Wilcoxon Signed-Rank 2 sample statistic
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.wilcoxon(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)


	def testTwoSampleKw(self, ds1, ds2, sigLev=.05):
		"""
		Kruskal-Wallis 2 sample statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.kruskal(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same distribution", "probably snot ame distribution", sigLev)

	def testTwoSampleFriedman(self, ds1, ds2, ds3, sigLev=.05):
		"""
		Friedman 2 sample statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		data3 = self.getNumericData(ds3)
		stat, pvalue = sta.friedmanchisquare(data1, data2, data3)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)

	def testTwoSampleEs(self, ds1, ds2, sigLev=.05):
		"""
		Epps Singleton 2 sample statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.epps_singleton_2samp(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)

	def testTwoSampleAnderson(self, ds1, ds2, sigLev=.05):
		"""
		Anderson 2 sample statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		dseq = (data1, data2)
		stat, critValues, sLev = sta.anderson_ksamp(dseq)
		slAlpha = 100 * sigLev

		if slAlpha == 10:
			cv = critValues[1]
		elif slAlpha == 5:
			cv = critValues[2]
		elif slAlpha == 2.5:
			cv = critValues[3]
		elif slAlpha == 1:
			cv = critValues[4]
		else:
			cv = None

		result = self.printResult("stat", stat, "critValues", critValues, "critValue", cv, "significanceLevel", sLev)
		print("stat:   {:.3f}".format(stat))
		if cv is None:
			msg = "critical values value not found for provided siginificance level"
		else:
			if stat < cv:
				msg = "probably same distribution at the {:.3f} siginificance level".format(sigLev)
			else:
				msg = "probably not same distribution at the {:.3f} siginificance level".format(sigLev)
		print(msg)
		return result


	def testTwoSampleScaleAb(self, ds1, ds2, sigLev=.05):
		"""
		Ansari Bradley 2 sample scale statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.ansari(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same scale", "probably not same scale", sigLev)
		return result

	def testTwoSampleScaleMood(self, ds1, ds2, sigLev=.05):
		"""
		Mood 2 sample scale statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.mood(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same scale", "probably not same scale", sigLev)
		return result

	def testTwoSampleVarBartlet(self, ds1, ds2, sigLev=.05):
		"""
		Ansari Bradley 2 sample scale statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.bartlett(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same variance", "probably not same variance", sigLev)
		return result

	def testTwoSampleVarLevene(self, ds1, ds2, sigLev=.05):
		"""
		Levene 2 sample variance statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.levene(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same variance", "probably not same variance", sigLev)
		return result

	def testTwoSampleVarFk(self, ds1, ds2, sigLev=.05):
		"""
		Fligner-Killeen 2 sample variance statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.fligner(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue)
		self.printStat(stat, pvalue, "probably same variance", "probably not same variance", sigLev)
		return result

	def testTwoSampleMedMood(self, ds1, ds2, sigLev=.05):
		"""
		Mood 2 sample median statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue, median, ctable = sta.median_test(data1, data2)
		result = self.printResult("stat", stat, "pvalue", pvalue, "median", median, "contigencyTable", ctable)
		self.printStat(stat, pvalue, "probably same median", "probably not same median", sigLev)
		return result

	def testTwoSampleZc(self, ds1, ds2, sigLev=.05):
		"""
		Zhang C 2 sample statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		l1 = len(data1)
		l2 = len(data2)
		l = l1 + l2
			
		#find ranks
		pooled = np.concatenate([data1, data2])
		ranks = findRanks(data1, pooled)
		ranks.extend(findRanks(data2, pooled))
		
		s1 = 0.0
		for i in range(1, l1+1):
			s1 += math.log(l1 / (i - 0.5) - 1.0) * math.log(l / (ranks[i-1] - 0.5) - 1.0)
			
		s2 = 0.0
		for i in range(1, l2+1):
			s2 += math.log(l2 / (i - 0.5) - 1.0) * math.log(l / (ranks[l1 + i - 1] - 0.5) - 1.0)
		stat = (s1 + s2) / l
		print(formatFloat(3, stat, "stat:"))
		return stat

	def testTwoSampleZa(self, ds1, ds2, sigLev=.05):
		"""
		Zhang A 2 sample statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		l1 = len(data1)
		l2 = len(data2)
		l = l1 + l2
		pooled = np.concatenate([data1, data2])
		cd1 = CumDistr(data1)
		cd2 = CumDistr(data2)
		sum = 0.0
		for i in range(1, l+1):
			v = pooled[i-1]
			f1 = cd1.getDistr(v)
			f2 = cd2.getDistr(v)
			
			t1 = f1 * math.log(f1)
			t2 = 0 if f1 == 1.0 else (1.0 - f1) * math.log(1.0 - f1)
			sum += l1 * (t1 + t2) / ((i - 0.5) * (l - i + 0.5))
			t1 = f2 * math.log(f2)
			t2 = 0 if f2 == 1.0 else (1.0 - f2) * math.log(1.0 - f2)
			sum += l2 * (t1 + t2) / ((i - 0.5) * (l - i + 0.5))
		stat = -sum
		print(formatFloat(3, stat, "stat:"))
		return stat

	def testTwoSampleZk(self, ds1, ds2, sigLev=.05):
		"""
		Zhang K 2 sample statistic	
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		l1 = len(data1)
		l2 = len(data2)
		l = l1 + l2
		pooled = np.concatenate([data1, data2])
		cd1 = CumDistr(data1)
		cd2 = CumDistr(data2)
		cd = CumDistr(pooled)
		
		maxStat = None
		for i in range(1, l+1):
			v = pooled[i-1]
			f1 = cd1.getDistr(v)
			f2 = cd2.getDistr(v)
			f = cd.getDistr(v)
			
			t1 = 0 if f1 == 0 else f1 * math.log(f1 / f)
			t2 = 0 if f1 == 1.0 else (1.0 - f1) * math.log((1.0 - f1) / (1.0 - f))
			stat = l1 * (t1 + t2)
			t1 = 0 if f2 == 0 else f2 * math.log(f2 / f)
			t2 = 0 if f2 == 1.0 else (1.0 - f2) * math.log((1.0 - f2) / (1.0 - f))
			stat += l2 * (t1 + t2)
			if maxStat is None or stat > maxStat:
				maxStat = stat
		print(formatFloat(3, maxStat, "stat:"))
		return maxStat


	def testTwoSampleCvm(self, ds1, ds2, sigLev=.05):
		"""
		2 sample cramer von mises
		"""
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		data = np.concatenate((data1,data2))
		rdata = sta.rankdata(data)
		n = len(data1)
		m = len(data2)
		l = n + m

		s1 = 0
		for i in range(n):
			t = rdata[i] - (i+1)	
			s1 += (t * t)
		s1 *= n

		s2 = 0
		for i in range(m):
			t = rdata[i] - (i+1)	
			s2 += (t * t)
		s2 *= m

		u = s1 + s2
		stat = u / (n * m * l) - (4 * m * n - 1) / (6 * l)
		result = self.printResult("stat", stat)
		return result

	def printStat(self, stat, pvalue, nhMsg, ahMsg, sigLev=.05):
		"""
		generic stat and pvalue output
		"""
		print("\ntest result:")
		print("stat:   {:.3f}".format(stat))
		print("pvalue: {:.3f}".format(pvalue))
		print("significance level: {:.3f}".format(sigLev))
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


