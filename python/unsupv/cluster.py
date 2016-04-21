#!/Users/pranab/Tools/anaconda/bin/python


# Package imports
import os
import sys
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import random
import jprops
from sklearn.externals import joblib
from random import randint
sys.path.append(os.path.abspath("../lib"))
from support import *

if len(sys.argv) < 2:
	print "usage: ./cluster.py <config_properties_file>"
	sys.exit()

# kmeans cluster finding
def train_kmeans():
	print "starting kmeans clustering..."
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

# agglomerative clustering
def train_agglomerative():
	print "starting agglomerative clustering..."
	model = AgglomerativeClustering(n_clusters=num_clusters, affinity=aggl_affinity,  
	linkage=aggl_linkage)
	model.fit(X)
	labels = model.labels_	
	print labels

# DBSCAN clustering
def train_dbscan():
	print "starting dbscan clustering..."
	model = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples, metric=dbs_metric, algorithm='auto')
	model.fit(X)
	
	core_ponts = model.core_sample_indices_ 
	if output_core_points:
		print "core points data index"
		print core_points
	print "num of core points %d" %(len(core_ponts))
	
	print "all points clutser index"
	cluster_index = model.labels_
	if output_cluster_members:
		#print cluster_index
		cluster_members = {}
		for i,c in enumerate(cluster_index):
			index_list = cluster_members.get(c, list())
			index_list.append(i)
			cluster_members[c] = index_list
		for cl, indx_list in cluster_members.iteritems():
			if cl > 0:
				print "cluster index %d  size %d" %(cl, len(indx_list))
			else:
				print "noise points size %d" %(len(indx_list))
			print indx_list
	
	print "num of clusters %d" %(cluster_index.max() + 1)
	
	
# finds belonging clusters
def predict():
	X = extract_data("pred.data.file", "pred.data.feature.fields")
	model_file_directory = configs["common.model.directory"]
	model_file_prefix = configs["common.model.file.prefix"]
	model_file = model_file_directory + "/" + model_file_prefix + ".mod"
	print "loading model file " +  model_file
	model = joblib.load(model_file) 	
	cluster_index = model.predict(X)
	print cluster_index

# calculates hopkins stat to find out whether data is likely to have clusters
def explore():
	print "calculating hopkins stats to detect if the data set has clusters..."
	X = extract_data("train.data.file", "train.data.feature.fields")	
	XR = extract_data("expl.data.file", "expl.data.feature.fields")
	split_size = int(configs["expl.sample.size"])
	num_iters = int(configs["expl.num.iters"])
	hopkins_stats = []
	
	for i in range(0, num_iters):
		(X_spl, X_tra) = split_data_random(X, split_size)
		(X_ran, X_rem) = split_data_random(XR, split_size)
	
		min_dist_ran = find_min_distances(X_ran, X_tra)
		min_dist_spl = find_min_distances(X_spl, X_tra)
	
		print "random"
		#print min_dist_ran
		ran_sum = min_dist_ran.sum()
		print "sum %.3f" %(ran_sum)
		
		print "split"
		#print min_dist_spl
		spl_sum = min_dist_spl.sum()
		print "sum %.3f" %(spl_sum)
	
		hopkins_stat = spl_sum / (ran_sum + spl_sum)
		print "hopkins stats %.3f" %(hopkins_stat)
		hopkins_stats.append(hopkins_stat)
		
	av_hopkins_stat = np.mean(hopkins_stats)
	print "average hopkins stat %.3f" %(av_hopkins_stat)

# loads file and extracts specific columns
def extract_data(file_name_param, field_indices_param):
	X = extract_table_from_file(configs, file_name_param, field_indices_param)	
	#preprocess features
	if (preprocess == "scale"):
		X = sk.preprocessing.scale(X)
	return X
	
# main
configs = get_configs(sys.argv[1])
mode = configs["common.mode"]
preprocess = configs["common.preprocessing"]

if mode == "train":
	#train
	print "running in training mode..."
	algo = configs["train.algo"]	
	num_clusters = int(configs["train.num.clusters"])
	num_iters = int(configs["train.num.iters"])
	if num_iters == -1:
		num_iters = 300
	persist_model = configs["train.persist.model"].lower() == "true"

	X = extract_data("train.data.file", "train.data.feature.fields")
	model_file_directory = configs["common.model.directory"]
	model_file_prefix = configs["common.model.file.prefix"]
	output_cluster_members = configs["train.output.cluster.members"].lower() == "true"
	
	if algo == "kmeans":
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
		train_kmeans()
	elif algo == "agglomerative":
		aggl_affinity = configs["train.affinity"]
		if aggl_affinity == "default":
			aggl_affinity = euclidean
		aggl_linkage = configs["train.linkage"]
		train_agglomerative()
	elif algo == "dbscan":
		dbs_eps = float(configs["train.eps"])
		if (dbs_eps < 0):
			dbs_eps = 0.5
		dbs_min_samples = int(configs["train.min.samples"])
		if dbs_min_samples < 0:
			dbs_min_samples = 5
		dbs_metric = configs["train.metric"]
		if dbs_metric == "default":
			dbs_metric = "euclidean"
		output_core_points = configs["train.output.core.points"].lower() == "true"
		train_dbscan()
	else:
		print "invalid cluster algorithm"
		sys.exit()

elif mode == "predict":
	print "running in prediction mode..."
	predict()
			
elif mode == "explore":
	#calculate hopkins stats
	print "running in explore mode..."
	explore()	
	
	
	
	
	
			