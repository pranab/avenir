#!/Users/pranab/Tools/anaconda/bin/python

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
 

import sys
import random 
import time
import math
import support 
import numpy as np
import math
from scipy.stats import norm

# gweke sample convergence for mc simulation
class GewekeConvergence:
	def __init__(self, burn_in_size_list):
		self.burn_in_size_list = burn_in_size_list
		self.zscores = []
		self.window_a = 0.1
		self.window_b = 0.5
		
	# modified z score	
	def calculate_zscore(self, data):
		n = len(data)
		for bi in self.burn_in_size_list:
			a_beg = bi
			a_end = bi + (n - bi) * self.window_a
			a = np.array(data[a_beg:a_end])
			b_beg = n - (n - bi) * self.window_b
			b = np.array(data[b_beg:])
			a_mean = a.mean()
			b_mean = b.mean()
			a_er = a.var() / len(a)
			b_er = b.var() / len(b)
			z_score = (a_mean - b_mean) / math.sqrt(a_er + b_er)
			self.zscores.append((n, bi, z_score))
			
	def get_zscores(self):
		return self.zscores
			
# Raftery Lewis  convergence
class RafteryLewisConvergence:
	def __init__(self, thinning_interval, percent_value_prob,  percent_value_conf_interval, trans_prob_conf_limit):
		# k:thinning_interval s:percent_value_prob r:percent_value_conf_interval e:trans_prob_conf_limit
		self.thinning_interval
		self.percent_value_prob = percent_value_prob
		self.percent_value_conf_interval = percent_value_conf_interval
		self.trans_prob_conf_limit = trans_prob_conf_limit
	
	# calculates burin in size and sample size	
	def find_sample_size(data):
		r = random.random().len(data)
		u = data[r]
		
		# z array based on threshold
		zdata = np.zeros(len(data))
		for i,d in enumerate(data):
			if d[i] < u:
				zdata[i] = 1;
			else:
				zdata[i] = 0
		
		# transition matrix
		tr = np.qeros(4)
		tr.shape = (2,2)
		for i,z in enumerate(zdata):
			if i > 0:
				cur = zdata[i-1]
				nex = z
				tr[cur][nex] = tr[cur][nex] + 1
		
		#normalize
		alpha = float(tr[0][1]) / (tr[0][0] + tr[0][1])
		beta = float(tr[1][0]) / (tr[1][0] + tr[1][1])
		
		# burn in size
		lambd = 1 - alpha - beta
		burn_in_size = math.log(self.trans_prob_conf_limit * (alpha + beta) / max(alpha, beta)) 
		burn_in_size /= math.log(lambd)
		burn_in_size *= self.thinning_interval
		
		
		# sample size
		samp_size = alpha * beta * (2 - alpha - beta) / (aplpha + beta) ** 3
		phi = norm.cdf(0.5 * (1 + self.percent_value_prob))
		samp_size /= (self.percent_value_conf_interval / phi) ** 2
		samp_size *= self.thinning_interval
		
		return (burn_in_size, samp_size)
		
		
