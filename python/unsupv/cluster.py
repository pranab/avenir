#!/Users/pranab/Tools/anaconda/bin/python

# Package imports
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.linear_model
from sklearn.cluster import KMeans
import matplotlib
import random
import jprops
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
from random import randint
sys.path.append(os.path.abspath("../lib"))
from support import *

if len(sys.argv) < 2:
	print "usage: ./cluster.py <config_properties_file>"
	sys.exit()

def train_kmeans():
	model = KMeans(n_clusters=num_clusters, init=init_strategy, n_init=num_inits, 
	max_iter=num_iters, precompute_distances=precom_dist)
	model.fit(X)
	
	#persist model
	if persist_model:
		model_file = model_file_directory + "/" + model_file_prefix + ".mod"
		print "saving model file " +  model_file
		joblib.dump(model, model_file) 
	
	clusters = model.cluster_centers_
	print clusters
	cohesion = model.inertia_ / len(X)
	print "cohesion:  %.3f" %(cohesion) 

configs = getConfigs(sys.argv[1])
mode = configs["common.mode"]
if mode == "train":
	#train
	preprocess = configs["common.preprocessing"]
	algo = configs["train.algo"]
	num_clusters = int(configs["train.num.clusters"])
	num_iters = int(configs["train.num.iters"])
	if num_iters == -1:
		num_iters = 300
	num_inits = int(configs["train.num.inits"])
	if num_inits == -1:
		num_inits = 10
	init_strategy = configs["train.init.strategy"]
	if init_strategy == "default":
		init_strategy = "k-means++"
	precom_dist = configs["train.precompute.distance"].lower()
	if precom_dist == "true":
		precom_dist = True
	elif precom_dist == "false":
		precom_dist = False
	elif not precom_dist == "auto":
		print "ivalid parameter for train.precompute.distance"
		sys.exit()
	persist_model = configs["train.persist.model"].lower() == "true"

	X = extract_table_from_file(configs, "train.data.file", "train.data.feature.fields")	
	#preprocess features
	if (preprocess == "scale"):
		X = sk.preprocessing.scale(X)
	model_file_directory = configs["common.model.directory"]
	model_file_prefix = configs["common.model.file.prefix"]
	
	if algo == "kmeans":
		train_kmeans()
		