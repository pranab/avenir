#!/Users/pranab/Tools/anaconda/bin/python

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

if len(sys.argv) < 2:
	print "usage: ./svm.py <config_properties_file>"
	sys.exit()


#linear k fold validation
def kfold_validation(nfold):
	model = build_model()
	#scores = sk.cross_validation.cross_val_score(model, XC, yc, cv=nfold)
	#print scores
	
	offset = 0
	length = dsize / nfold
	errors = []
	for i in range(0, nfold):
		print "....Next fold %d" %(i)
		
		#split data
		(XV,yv,X,y) = split_data(offset, length)
		dvsize = len(XV)

		#train model
		model.fit(X, y) 

		#persist model
		model_file = model_file_prefix + "_" + str(i + 1) + ".mod"
		joblib.dump(model, model_file) 
		
		#print support vectors
		print_support_vectors(model)
		
		#predict
		print "making predictions..."
		yp = model.predict(XV)

		#show prediction output
		error = validate(dvsize,yv,yp)
		errors.append(error)
		
		offset += length
		
	#average error
		
		
# random k fold validation
def rfold_validation(nfold, niter):
	max_offset_frac = 1.0 - 1.0 / nfold
	max_offset_frac -= .01
	length = dsize / nfold

	errors = []
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
		model_file = model_file_prefix + "_" + str(i + 1) + ".mod"
		joblib.dump(model, model_file) 

		#print support vectors
		print_support_vectors(model)

		#predict
		print "making predictions..."
		yp = model.predict(XV)

		#show prediction output
		error = validate(dvsize,yv,yp)
		errors.append(error)
		
	av_error = np.mean(errors)
	print "average error %.3f" %(av_error)
	
#builds model	
def build_model():	
	#build model
	print "building model..."
	if (algo == "svc"):
		model = sk.svm.SVC(C=penalty,kernel=kernel_fun)
	elif (algo == "nusvc"):
		model = sk.svm.NuSVC(kernel=kernel_fun)
	elif (algo == "linearsvc"):
		model = sk.svm.LinearSVC(C=penalty)
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
		
	er = float(err_count)  / dvsize		
	print "error %.3f" %(er)
	print "true positive : %.3f" %(float(tp)  / dvsize)
	print "false positive: %.3f" %(float(fp)  / dvsize)
	print "true negative : %.3f" %(float(tn)  / dvsize)
	print "false negative: %.3f" %(float(fn)  / dvsize)

	return er

# load configuration
def getConfigs(configFile):
	configs = {}
	with open(configFile) as fp:
  		for key, value in jprops.iter_properties(fp):
			print key, value
			configs[key] = value

	return configs
	
######################################################################
configs = getConfigs(sys.argv[1])
data_file = configs["train.data.file"]
feat_field_indices = configs["train.data.feature.fields"].split(",")
feat_field_indices = [int(a) for a in feat_field_indices]
class_field_index = int(configs["train.data.class.field"])
preprocess = configs["train.preprocessing"]
validation = configs["train.validation"]
num_folds = int(configs["train.num.folds"])
num_iter = int(configs["train.num.iter"])
algo = configs["train.algorithm"]
kernel_fun = configs["train.kernel.function"]
penalty = float(configs["train.penalty"])
if penalty is None:
	penalty = 1.0
print_sup_vectors = configs["train.print.sup.vectors"].lower() == "true"
persist_model = configs["train.persist.model"].lower() == "true"
model_file_prefix = configs["train.model.file.prefix"]
	
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
	kfold_validation(num_folds)
else:
	rfold_validation(num_folds,num_iter)
	
	
	