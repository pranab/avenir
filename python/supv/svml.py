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
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.linear_model
import matplotlib
import random
import jprops
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
from random import randint

if len(sys.argv) < 2:
	print "usage: ./svm.py <config_properties_file>"
	sys.exit()

#train by bagging
def train_bagging():
	model = build_model()
	bagging_model = BaggingClassifier(base_estimator=model,n_estimators=bagging_num_estimator,
	max_samples=bagging_sample_fraction,oob_score=bagging_use_oob)
	
	#train model
	bagging_model.fit(XC, yc) 
	
	#persist model
	if persist_model:
		models = bagging_model.estimators_
		for m in zip(range(0, len(models)), models):
			model_file = model_file_directory + "/" + model_file_prefix + "_" + str(m[0] + 1) + ".mod"
			joblib.dump(m[1], model_file) 

	score = bagging_model.score(XC, yc)
	print "average error %.3f" %(1.0 - score)

#linear k fold validation
def train_kfold_validation(nfold):
	if native_kfold_validation:
		print "native linear kfold validation"
		model = build_model()
		scores = sk.cross_validation.cross_val_score(model, XC, yc, cv=nfold)
		av_score = np.mean(scores)
		print "average error %.3f" %(1.0 - av_score)
	else:
		print "extended linear kfold validation"
		train_kfold_validation_ext(nfold)

#linear k fold validation
def train_kfold_validation_ext(nfold):
	model = build_model()
	#scores = sk.cross_validation.cross_val_score(model, XC, yc, cv=nfold)
	#print scores
	
	offset = 0
	length = dsize / nfold
	errors = []
	fp_errors = []
	fn_errors = []
	for i in range(0, nfold):
		print "....Next fold %d" %(i)
		
		#split data
		(XV,yv,X,y) = split_data(offset, length)
		dvsize = len(XV)

		#train model
		model.fit(X, y) 

		#persist model
		if persist_model:
			model_file = model_file_directory + "/" + model_file_prefix + "_" + str(i + 1) + ".mod"
			joblib.dump(model, model_file) 
		
		#print support vectors
		print_support_vectors(model)
		
		#predict
		print "making predictions..."
		yp = model.predict(XV)

		#show prediction output
		(er, fp_er, fn_er) = validate(dvsize,yv,yp)
		errors.append(er)
		fp_errors.append(fp_er)
		fn_errors.append(fn_er)
		
		offset += length
		
	#average error
	av_error = np.mean(errors)
	av_fp_error = np.mean(fp_errors)
	av_fn_error = np.mean(fn_errors)
	print "average  error %.3f  false positive error %.3f  false negative error %.3f" %(av_error, av_fp_error, av_fn_error)

# random k fold validation
def train_rfold_validation(nfold, niter):
	if native_rfold_validation:
		print "native random  kfold validation"
		train_fraction = 1.0 / nfold
		scores = []
		for i in range(0,niter):
			state = randint(1,100)
			X, XV, y, yv = sk.cross_validation.train_test_split(XC, yc, test_size=train_fraction, random_state=state)
			model = build_model()
			model.fit(X,y)
			scores.append(model.score(XV, yv))
		
		print scores
		av_score = np.mean(scores)
		print "average error %.3f" %(1.0 - av_score)

	else:
		print "extended random  kfold validation"
		train_rfold_validation_ext(nfold, niter)
		
# random k fold validation
def train_rfold_validation_ext(nfold, niter):
	max_offset_frac = 1.0 - 1.0 / nfold
	max_offset_frac -= .01
	length = dsize / nfold

	errors = []
	fp_errors = []
	fn_errors = []
	for i in range(0,niter):	
		print "...Next iteration %d" %(i)
		offset = int(dsize * random.random() * max_offset_frac)
		print "offset: %d  length: %d" %(offset, length)
		(XV,yv,X,y) = split_data(offset, length)
		dvsize = len(XV)
	
		#build model
		model = build_model()
	
		#train model
		model.fit(X, y) 
		
		#persist model
		if persist_model:
			model_file = model_file_directory + "/" + model_file_prefix + "_" + str(i + 1) + ".mod"
			print "saving model file " +  model_file
			joblib.dump(model, model_file) 

		#print support vectors
		print_support_vectors(model)

		#predict
		print "making predictions..."
		yp = model.predict(XV)

		#show prediction output
		(er, fp_er, fn_er) = validate(dvsize,yv,yp)
		errors.append(er)
		fp_errors.append(fp_er)
		fn_errors.append(fn_er)
		
	av_error = np.mean(errors)
	av_fp_error = np.mean(fp_errors)
	av_fn_error = np.mean(fn_errors)
	print "average error %.3f  false positive error %.3f  false negative error %.3f" %(av_error, av_fp_error, av_fn_error)

# make predictions
def predict():
	psize = len(X)
	class_counts = []
	
	#all models
	for i in range(0, num_models):
		model_file = model_file_directory + "/" + model_file_prefix + "_" + str(i + 1) + ".mod"
		print "loading model file " +  model_file
		model = joblib.load(model_file) 	
		
		yp = model.predict(X)
		if i == 0:
			#initialize class counts
			for y in yp:
				class_count = {}
				if y == 0:
					class_count[0] = 1
					class_count[1] = 0
				else:	
					class_count[1] = 1
					class_count[0] = 0
				class_counts.append(class_count)
				
		else:
			#increment class count
			for j in range(0, psize):
				class_count = class_counts[j]
				y = yp[j]
				class_count[y] +=  1
	
	# predict based on majority vote
	print "here are the predictions"
	for k in range(0, psize):
		class_count = class_counts[k]
		if (class_count[0] > class_count[1]):
			y = 0
			majority = class_count[0]
		else:
			y = 1
			majority = class_count[1]
			
		print X[k]
		print "prediction %d  majority count %d" %(y, majority)
		
#builds model	
def build_model():	
	#build model
	print "building model..."
	if algo == "svc":
		if kernel_fun == "poly":
			model = sk.svm.SVC(C=penalty,kernel=kernel_fun,degree=poly_degree,gamma=kernel_coeff)
		elif kernel_fun == "rbf" or kernel_fun == "sigmoid":
			model = sk.svm.SVC(C=penalty,kernel=kernel_fun,gamma=kernel_coeff)
		else:
			model = sk.svm.SVC(C=penalty,kernel=kernel_fun)
	elif algo == "nusvc":
		if kernel_fun == "poly":
			model = sk.svm.NuSVC(kernel=kernel_fun,degree=poly_degree,gamma=kernel_coeff)
		elif kernel_fun == "rbf" or kernel_fun == "sigmoid":
			model = sk.svm.NuSVC(kernel=kernel_fun,gamma=kernel_coeff)
		else:
			model = sk.svm.NuSVC(kernel=kernel_fun)
	elif algo == "linearsvc":
		model = sk.svm.LinearSVC()
	else:
		print "invalid svm algorithm"
		sys.exit()
	return model

#splits data into training and validation sets	
def split_data(offset, length):
	print "splitting data..."
	#copy data
	XC_c = np.copy(XC)
	yc_c = list(yc)
	
	# validation set
	vlo = offset
	vup = vlo + length
	if (vup > len(yc)):
		vup = len(yc)
	XV = XC_c[vlo:vup:1]
	yv = yc_c[vlo:vup:1]
	dvsize = len(XV)
	print "data size %d validation data size %d" %(dsize, dvsize)
	#print "validation set"
	#print XV
	#print yv

	#training set
	X = np.delete(XC_c, np.s_[vlo:vup:1], 0)
	y = np.delete(yc_c, np.s_[vlo:vup:1], 0)
	#print "training set"
	#print X
	#print y
	return (XV,yv,X,y)

#print support vectors
def print_support_vectors(model):
	if (not algo == "linearsvc"):
		if print_sup_vectors:
			print "showing support vectors..." 
			print model.support_vectors_
		print "num of support vectors"
		print model.n_support_

#prints prediction output
def validate(dvsize,yv,yp):
	print "showing predictions..."
	err_count = 0
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for r in range(0,dvsize):
		#print "actual: %d  predicted: %d" %(yv[r], yp[r])
		if (not yv[r] ==  yp[r]):
			err_count += 1
			
		if (yp[r] == 1 and yv[r] == 1):
			tp += 1
		elif (yp[r] == 1 and yv[r] == 0):
			fp += 1
		elif (yp[r] == 0 and yv[r] == 0):
			tn += 1
		else:
			fn += 1
		
	er = float(err_count) / dvsize		
	fp_er = float(fp) / dvsize
	fn_er = float(fn) / dvsize
	print "error %.3f" %(er)
	print "true positive : %.3f" %(float(tp) / dvsize)
	print "false positive: %.3f" %(fp_er)
	print "true negative : %.3f" %(float(tn) / dvsize)
	print "false negative: %.3f" %(fn_er)

	return (er, fp_er, fn_er)

# load configuration
def getConfigs(configFile):
	configs = {}
	print "using following configurations"
	with open(configFile) as fp:
  		for key, value in jprops.iter_properties(fp):
			print key, value
			configs[key] = value

	return configs
	

# load configuration
configs = getConfigs(sys.argv[1])
mode = configs["common.mode"]

if mode == "train":
	#train
	print "running in train mode"
	data_file = configs["train.data.file"]
	feat_field_indices = configs["train.data.feature.fields"].split(",")
	feat_field_indices = [int(a) for a in feat_field_indices]
	class_field_index = int(configs["train.data.class.field"])
	preprocess = configs["common.preprocessing"]
	validation = configs["train.validation"]
	num_folds = int(configs["train.num.folds"])
	num_iter = int(configs["train.num.iter"])
	algo = configs["train.algorithm"]
	kernel_fun = configs["train.kernel.function"]
	poly_degree = int(configs["train.poly.degree"])
	penalty = float(configs["train.penalty"])
	if penalty < 0:
		penalty = 1.0
		print "using default for penalty"
	kernel_coeff = float(configs["train.gamma"])
	if kernel_coeff < 0:
		kernel_coeff = 'auto'
		print "using default for gamma"
	print_sup_vectors = configs["train.print.sup.vectors"].lower() == "true"
	persist_model = configs["train.persist.model"].lower() == "true"
	model_file_directory = configs["common.model.directory"]
	model_file_prefix = configs["common.model.file.prefix"]
	
	print feat_field_indices
	
	#extract feature fields
	d = np.loadtxt(data_file, delimiter=',')
	dsize = len(d)
	XC = d[:,feat_field_indices]

	#preprocess features
	if (preprocess == "scale"):
		XC = sk.preprocessing.scale(XC)
	elif (preprocess == "normalize"):
		XC = sk.preprocessing.normalize(XC, norm='l2')
	else:
		print "no preprocessing done"

	#extract output field
	yc = d[:,[class_field_index]]
	yc = yc.reshape(dsize)
	yc = [int(a) for a in yc]

	#print XC
	#print yc
	
	
	# train model
	if validation == "kfold":
		native_kfold_validation = configs["train.native.kfold.validation"].lower() == "true"
		train_kfold_validation(num_folds)
	elif validation == "rfold":
		native_rfold_validation = configs["train.native.rfold.validation"].lower() == "true"
		train_rfold_validation(num_folds,num_iter)
	elif validation == "bagging":
		bagging_num_estimator = int(configs["train.bagging.num.estimators"])
		bagging_sample_fraction = float(configs["train.bagging.sample.fraction"])
		bagging_use_oob = configs["train.bagging.sample.fraction"].lower() == "true"
		train_bagging()
	else:
		print "invalid training validation method"
		sys.exit()
		
else:
	#predict
	print "running in prediction mode"
	pred_data_file = configs["pred.data.file"]
	pred_feat_field_indices = configs["pred.data.feature.fields"].split(",")
	pred_feat_field_indices = [int(a) for a in pred_feat_field_indices]
	preprocess = configs["common.preprocessing"]
	num_models = int(configs["pred.num.models"])
	model_file_directory = configs["common.model.directory"]
	model_file_prefix = configs["common.model.file.prefix"]
	
	#extract feature fields
	pd = np.loadtxt(pred_data_file, delimiter=',')
	pdsize = len(pd)
	X = pd[:,pred_feat_field_indices]
	
	#preprocess features
	if (preprocess == "scale"):
		X = sk.preprocessing.scale(X)
	elif (preprocess == "normalize"):
		X = sk.preprocessing.normalize(X, norm='l2')
	else:
		print "no preprocessing done"
	
	predict()
