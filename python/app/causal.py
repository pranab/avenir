#!/usr/local/bin/python3

# beymani-python: Machine Learning
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
import random
import statistics 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import cdt
import networkx as nx
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
sys.path.append(os.path.abspath("../supv"))
from util import *
from sampler import *
from daexp import *

"""
causality analysis 
"""

def getEntropy(expl, d1, d2, algo="norm"):
	"""
	
	"""
	if algo == "norm":
		e1 = expl.getEntropy(d1, 20)
		e2 = expl.getEntropy(d2, 20)
	elif algo == "onsp":
		e1 = expl.oneSpaceEntropy(d1)
		e2 = expl.oneSpaceEntropy(d2)
	
	r = (e1, e2)
	return r
		
if __name__ == "__main__":
	op = sys.argv[1]
	if op == "disc":
		"""  pair wise causal discovery with additive noise model """
		dist1 = NormalSampler(1200, 10)	
		dist2 = NormalSampler(800, 3)
		noise = NormalSampler(0, 60)
		noiseX = NormalSampler(0, 20)
	
		d1 = list()
		d2 = list()
		qu = 1200
		for i in range(100):
			#s = randomFloat(8,12) + noiseX.sample()
			#qu += s
			
			delta = i * randomFloat(6,10) 
			qu = dist1.sample() + delta
			
			d1.append(qu)
			d2.append(2000 - qu + noise.sample())
		
		 
		expl = DataExplorer(False)
		expl.addListNumericData(d1, "d1")
		expl.addListNumericData(d2, "d2")

		#expl.plotHist("d1", False, True)
		#expl.plotHist( "d2", False, True)

		print("** forward case")
		re = expl.fitLinearReg("d1", "d2")
		sl = re["slope"]
		intc = re["intercept"]
		nd1 = np.array(d1)
		nd2 = np.array(d2)
		#expl.plotRegFit(nd1, nd2, sl, intc)
		
		re = expl.getRegFit(d1, d2, sl, intc)
		res = re["residue"]
		expl.addListNumericData(res, "res")
		
		ex, er = getEntropy(expl, "d1", "res", algo="onsp") 
		e1 = ex["entropy"]
		e2 = er["entropy"]
		#print("{:.6f} {:.6f}".format(e1,e2))
		print("total entropy {:.6f}".format(ex["entropy"] + er["entropy"]))
		
		print("** reverse case")
		re = expl.fitLinearReg("d2", "d1")
		sl = re["slope"]
		intc = re["intercept"]
		nd1 = np.array(d1)
		nd2 = np.array(d2)
		#expl.plotRegFit(nd2, nd1, sl, intc)
		
		re = expl.getRegFit(d2, d1, sl, intc)
		res = re["residue"]
		expl.addListNumericData(res, "res")

		ex, er = getEntropy(expl, "d2", "res", algo="onsp") 
		e1 = ex["entropy"]
		e2 = er["entropy"]
		#print("{:.6f} {:.6f}".format(e1,e2))
		print("total entropy {:.6f}".format(ex["entropy"] + er["entropy"]))

	elif op == "ges":
		"""   causal graph discovery with GES """
		dataFilePath = sys.argv[2]
		cols = strListOrRangeToIntArray(sys.argv[3])
		df = pd.read_csv(dataFilePath, header=None, usecols=cols)
		
		glasso = cdt.independence.graph.Glasso()
		skeleton = glasso.predict(df)
		print("co variance")
		print(nx.adjacency_matrix(skeleton).todense())
		
		#remove indirect links
		newSkeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg="aracne")
		print("after removing indirect dependency")
		print(nx.adjacency_matrix(newSkeleton).todense())
		
		#discovery
		model = cdt.causality.graph.GES()
		outputGraph = model.predict(df, newSkeleton)
		print("causal graph")
		print(nx.adjacency_matrix(outputGraph).todense())
		
