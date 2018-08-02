#!/usr/bin/python

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
from random import randrange
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

# sample
def sample(data_file, wt_index):
	# build sampler
	fp = open(data_file, "r")
	values = []
	for line in fp:
		items = line.split(",")
		weight = double(items[wt_index])
		values.append(weight)
	fp.close()
	sampler = new NonParamRejectSampler(0, 1, values)
	
	# sample
	sampled_indices = []
	size = len(values)
	for i in range(size):
		sampled_indices.append(sampler.sample())
	sampled_indices.sort()
		
	# output
	fp = open(data_file, "r")
	max_weight = 0
	i = 0
	j = 0
	for line in fp:
		while sampled_indices[i] == j:
			print line
			i = i + 1 
		j = j + 1
		

# main   
data_file = sys.argv[1]
id_index = int(sys.argv[2])
wt_index = int(sys.argv[3]

sample(data_file, wt_index)

