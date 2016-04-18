#!/Users/pranab/Tools/anaconda/bin/python

# Package imports
import os
import sys
import numpy as np
import jprops
from random import randint

# load configuration
def getConfigs(configFile):
	configs = {}
	print "using following configurations"
	with open(configFile) as fp:
  		for key, value in jprops.iter_properties(fp):
			print key, value
			configs[key] = value

	return configs

# extract tabular data from csv file
def extract_table_from_file(configs, file_param, col_indices_param):
	data_file = configs[file_param]
	col_indices = configs[col_indices_param].split(",")
	col_indices = [int(a) for a in col_indices]
	data = np.loadtxt(data_file, delimiter=',')
	tabl = data[:,col_indices]
	return tabl
	
# finds minimum distance between each row f X1(m x p) and X2(n x p)
# return (1 x m) array of min distances
def find_min_distances(X1, X2):
	min_dist = np.zeros(len(X1))
	
	for i,x1 in enumerate(X1):
		dists = np.sqrt(np.sum((X2 - x1)**2,axis=1))
		min_dist[i] = dists.min()

	return min_dist

# splits data randomly to create two arrays	
def split_data_random(X, split_size):
	#copy data
	XC = np.copy(X)
	
	# split
	lo = randint(1, len(X) - split_size) - 1
	up = lo + split_size	
	XSP = XC[lo:up:1]
	

	#remaining
	XRE = np.delete(XC, np.s_[lo:up:1], 0)
	return (XSP, XRE)
