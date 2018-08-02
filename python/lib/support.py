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

# Package imports
import os
import sys
import numpy as np
import jprops
from random import randint

# load configuration
def get_configs(configFile):
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

# finds minimum distance between each row f X(n x p) and other rows in X
# returns (1 x n-1) array
def find_min_distances_between_rows(X):
	num_rows = X.shape[0] - 1
	min_dist = np.zeros(num_rows)
	
	for i,x1 in enumerate(X):
		row_min_dist = []
		for j,x2 in enumerate(X):
			if j > i:
				# upper diagonal only
				dist = np.sqrt(np.sum((x1 - x2)**2))
				row_min_dist.append(dist)
		if i < num_rows: 
			min_dist[i] = min(row_min_dist)

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

# min max scaling between 0 and 1
def scale_min_max(arr):
	min = np.min(arr)
	max = np.max(arr)
	new_arr = (arr - min) / (max - min)
	return new_arr
	
