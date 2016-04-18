#!/Users/pranab/Tools/anaconda/bin/python

# Package imports
import os
import sys
import numpy as np
import jprops

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
