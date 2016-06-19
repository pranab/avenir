#!/Users/pranab/Tools/anaconda/bin/python

import sys
import random 
import time
import math
import support 
import numpy as np
import math

class GewekeConvergence:
	def __init__(self, burn_in_size_list):
		self.burn_in_size_list = burn_in_size_list
		self.zscores = []
		self.window_a = 0.1
		self.window_b = 0.5
		
		
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
			
