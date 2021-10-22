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
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.neighbors import KDTree
import matplotlib
import random
import jprops
from random import randint
import statistics
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from tnn import *
from stats import *

"""
neural model calibration
"""
class ModelCalibration(object):
	def __init__(self):
		pass
		
	@staticmethod
	def findModelCalibration(model):
		"""
		pmodel calibration
		"""
		FeedForwardNetwork.prepValidate(model)
		FeedForwardNetwork.validateModel(model)
		
		yPred = model.yPred.flatten()
		yActual = model.validOutData.flatten()
		nsamp = len(yActual)
		
		#print(yPred.shape)
		#print(yActual.shape)
		
		nBins = model.config.getIntConfig("calibrate.num.bins")[0]
		prThreshhold = model.config.getFloatConfig("calibrate.pred.prob.thresh")[0]
		
		minConf = yPred.min()
		maxConf = yPred.max()
		bsize = (maxConf - minConf) / nBins
		#print("minConf {:.3f}  maxConf {:.3f}  bsize {:.3f}".format(minConf, maxConf, bsize))
		blist = list(map(lambda i : None, range(nBins)))
		
		#binning
		for yp, ya in zip(yPred, yActual):
			indx = int((yp - minConf) / bsize)
			if indx == nBins:
				indx = nBins - 1
			#print("yp {:.3f}  indx {}".format(yp, indx))
			pair = (yp, ya)
			plist  = blist[indx]
			if plist is None:
				plist = list()
				blist[indx] = plist 
			plist.append(pair)
		 
		x = list()
		y = list()
		yideal = list()
		ece = 0
		mce = 0
		
		# per bin confidence and accuracy
		b = 0
		for plist in blist:
			if plist is not None:
				#confidence
				ypl = list(map(lambda p : p[0], plist))
				ypm = statistics.mean(ypl)
				x.append(ypm)
			
				#accuracy
				ypcount = 0
				for p in plist:
					yp = 1 if p[0] > prThreshhold else 0
					if (yp == 1 and p[1] == 1):
						ypcount += 1
				 
				acc = ypcount / len(plist)
				y.append(acc)
				yideal.append(ypm)
			
				ce = abs(ypm - acc)
				ece += len(plist) * ce
				if ce > mce:
					mce = ce
			else:
				ypm = minConf + (b + 0.5) * bsize
				x.append(ypm)
				yideal.append(ypm)
				y.append(0)
			b += 1
				
		#calibration plot	
		drawPairPlot(x, y, yideal, "confidence", "accuracy", "actual", "ideal")
		
		print("confidence\taccuracy")
		for z in zip(x,y):
			print("{:.3f}\t{:.3f}".format(z[0], z[1]))
		
		
		#expected calibration error
		ece /= nsamp
		print("expected calibration error\t{:.3f}".format(ece))
		print("maximum calibration error\t{:.3f}".format(mce))
	
	
	@staticmethod
	def findModelCalibrationLocal(model):
		"""
		pmodel calibration based k nearest neghbors
		"""
		FeedForwardNetwork.prepValidate(model)
		FeedForwardNetwork.validateModel(model)
		
		yPred = model.yPred.flatten()
		yActual = model.validOutData.flatten()
		nsamp = len(yActual)
		
		neighborCnt =  model.config.getIntConfig("calibrate.num.nearest.neighbors")[0]
		prThreshhold = model.config.getFloatConfig("calibrate.pred.prob.thresh")[0]
		fData = model.validFeatData.numpy()
		tree = KDTree(fData, leaf_size=4)
		
		dist, ind = tree.query(fData, k=neighborCnt)
		calibs = list()
		#all data
		for si, ni in enumerate(ind):
			conf = 0
			ypcount = 0
			#all neighbors
			for i in ni:
				conf += yPred[i]
				yp = 1 if yPred[i] > prThreshhold else 0
				if (yp == 1 and yActual[i] == 1):
					ypcount += 1
			conf /= neighborCnt
			acc = ypcount / neighborCnt
			calib = (si, conf, acc)
			calibs.append(calib)
			
		#descending sort by difference between confidence and accuracy
		calibs = sorted(calibs, key=lambda c : abs(c[1] - c[2]), reverse=True)
		print("local calibration")
		print("conf\taccu\trecord")
		for i in range(19):
			si, conf, acc = calibs[i]
			rec = toStrFromList(fData[si], 3)
			print("{:.3f}\t{:.3f}\t{}".format(conf, acc, rec))

	@staticmethod
	def findModelSharpness(model):
		"""
		pmodel calibration
		"""
		FeedForwardNetwork.prepValidate(model)
		FeedForwardNetwork.validateModel(model)
		
		yPred = model.yPred.flatten()
		yActual = model.validOutData.flatten()
		nsamp = len(yActual)
		
		#print(yPred.shape)
		#print(yActual.shape)
		
		nBins = model.config.getIntConfig("calibrate.num.bins")[0]
		prThreshhold = model.config.getFloatConfig("calibrate.pred.prob.thresh")[0]
		
		minConf = yPred.min()
		maxConf = yPred.max()
		bsize = (maxConf - minConf) / nBins
		#print("minConf {:.3f}  maxConf {:.3f}  bsize {:.3f}".format(minConf, maxConf, bsize))
		blist = list(map(lambda i : None, range(nBins)))
		
		#binning
		for yp, ya in zip(yPred, yActual):
			indx = int((yp - minConf) / bsize)
			if indx == nBins:
				indx = nBins - 1
			#print("yp {:.3f}  indx {}".format(yp, indx))
			pair = (yp, ya)
			plist  = blist[indx]
			if plist is None:
				plist = list()
				blist[indx] = plist 
			plist.append(pair)
		 
		y = list()
		ypgcount = 0
		# per bin confidence and accuracy
		for plist in blist:
			#ypl = list(map(lambda p : p[0], plist))
			#ypm = statistics.mean(ypl)
			#x.append(ypm)
			
			ypcount = 0
			for p in plist:
				yp = 1 if p[0] > prThreshhold else 0
				if (yp == 1 and p[1] == 1):
					ypcount += 1
					ypgcount += 1
				 
			acc = ypcount / len(plist)
			y.append(acc)
		
		print("{} {}".format(ypgcount, nsamp))	
		accg = ypgcount / nsamp
		accgl = [accg] * nBins
		x = list(range(nBins))	
		drawPairPlot(x, y, accgl, "discretized confidence", "accuracy", "local", "global")
		
		contrast = list(map(lambda acc : abs(acc - accg), y))
		contrast = statistics.mean(contrast)
		print("contrast {:.3f}".format(contrast))

"""
neural model robustness
"""
class ModelRobustness(object):
	def __init__(self):
		pass
		
	def localPerformance(self, model, fpath, nsamp, neighborCnt):
		"""
		local performnance sampling
		"""
		
		#load data
		fData, oData = FeedForwardNetwork.prepData(model, fpath)
		#print(type(fData))
		#print(type(oData))
		#print(fData.shape)
		dsize = fData.shape[0]
		ncol = fData.shape[1]
		
		#kdd 
		tree = KDTree(fData, leaf_size=4)		

		scores = list()
		for _ in range(nsamp):
			indx = randomInt(0, dsize - 1)
			frow = fData[indx]
			frow = np.reshape(frow, (1, ncol))
			dist, ind = tree.query(frow, k=neighborCnt)
			
			ind = ind[0]
			vfData = fData[ind]	
			voData = oData[ind]	
			
			#print(type(vfData))
			#print(vfData.shape)
			#print(type(voData))
			#print(voData.shape)
			
			model.setValidationData((vfData, voData), False)
			score = FeedForwardNetwork.validateModel(model)
			scores.append(score)
		
		m, s = basicStat(scores)
		print("model performance:   mean {:.3f}\tstd dev {:.3f}".format(m,s))	
		drawHist(scores, "model accuracy", "accuracy", "frequency")
			
		