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
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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
def train_kmeans(num_clusters, init_strategy, num_inits, num_iters, precom_dist, 
model_file_directory, model_file_prefix):
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

	inter_cluster_distances = find_min_distances_between_rows(clusters)
	print inter_cluster_distances
	min_inter_cluster_distance = inter_cluster_distances.min()
	print "min inter cluster distance: %.3f" %(min_inter_cluster_distance)
	
	return (cohesion, num_clusters/min_inter_cluster_distance)

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
def expl_hopkins(configs):
	print "calculating hopkins stats to detect if the data set has clusters..."
	X = extract_data("train.data.file", "train.data.feature.fields")	
	XR = extract_data("expl.hopkins.data.file", "expl.hopkins.data.feature.fields")
	split_size = int(configs["expl.hopkins.sample.size"])
	num_iters = int(configs["expl.hopkins.num.iters"])
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


# calculates distance to nearest k th neighbor
def expl_kdist(configs):
	print "calculating distance to nearest kth neighbor ..."
	X = extract_data("train.data.file", "train.data.feature.fields")
	neighbor_index = int(configs["expl.kdist.neighbor.index"])	
	output_first_order_diff = configs["expl.kdist.output.dist.first.order.diff"].lower() == "true"
	
	neigh = NearestNeighbors(n_neighbors=neighbor_index)
	neigh.fit(X)
	dist = neigh.kneighbors(return_distance=True)[0]
	#print dist
	print "after sorting"
	dist.sort(axis=0)
	#print dist
	for k in range(0, neighbor_index):
		dist_kth = dist[:,k]
		print "sorted distance to nearest neighbor at position %d " %(k)
		print dist_kth	
		if output_first_order_diff:
			print "1st order diff of sorted distance to kth neighbor"
			diff_dist_kth = np.diff(dist_kth)
			print diff_dist_kth

# loads file and extracts specific columns
def extract_data(file_name_param, field_indices_param):
	X = extract_table_from_file(configs, file_name_param, field_indices_param)	
	#preprocess features
	if (preprocess == "scale"):
		X = sk.preprocessing.scale(X)
	return X

def validity_index(un_part, ov_part):	
	un_part = scale_min_max(un_part)
	ov_part = scale_min_max(ov_part)
	validity = un_part + ov_part
	return validity
	
# main
configs = get_configs(sys.argv[1])
mode = configs["common.mode"]
preprocess = configs["common.preprocessing"]

if mode == "train":
	#train
	print "running in training mode..."
	algo = configs["train.algo"]
	num_cluster_list = 	configs["train.num.clusters"].split(",")
	num_cluster_list = [int(a) for a in num_cluster_list]
	num_clusters = num_cluster_list[0]
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

		num_partition_tries = len(num_cluster_list)		
		if num_partition_tries == 1:
			train_kmeans(num_cluster_list[0], init_strategy, num_inits, num_iters, precom_dist, 
			model_file_directory, model_file_prefix)
		else:
			un_part = np.zeros(num_partition_tries)	
			ov_part = np.zeros(num_partition_tries)	
			for i,num_clusters in enumerate(num_cluster_list):
				print "starting with num of clusters %d ..." %(num_clusters)
				result = train_kmeans(num_clusters, init_strategy, num_inits, num_iters, precom_dist, 
				model_file_directory, model_file_prefix)
				print "partition measure"
				print result
				un_part[i] = result[0]
				ov_part[i] = result[1]
				
			#validity index
			validity = validity_index(un_part, ov_part)
			print "validity index"
			for c,v in zip(num_cluster_list, validity):
				print "num cluster %d validity %.3f" %(c, v)
	
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
	expl_algo = configs["expl.algo"]
	if 	expl_algo == "hopkins":
		expl_hopkins(configs)
	
	elif expl_algo == "kdist":
		expl_kdist(configs)
	
	else:
		print "invalid cluster exploration algorithm"
		sys.exit()
	
	
	
	
			
