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
import jprops
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

	def plot(self, dname, yscale=None):
		"""
		plots data
		"""
		assert dname in self.dataSets, "data set {} does not exist".format(dname)
		data = self.dataSets[dname]
		drawLine(data, yscale)

	def difference(self, dname, order):
		"""
		difference of given order
		"""
		assert dname in self.dataSets, "data set {} does not exist".format(dname)
		data = self.dataSets[dname]
		diff = difference(data, order)
		drawLine(diff)
		return diff


