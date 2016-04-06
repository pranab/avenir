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


if len(sys.argv) < 7:
	print "usage: <training_data_file> <index_of_feature_fields> <index_of_class_field> <preprocess> <validation> <algorithm> <kernel> <penalty>"
	sys.exit()

data_file = sys.argv[1]
feat_field_indices = sys.argv[2].split(",")
feat_field_indices = [int(a) for a in feat_field_indices]
class_field_index = int(sys.argv[3])
preprocess = sys.argv[4]
validation = sys.argv[5]

algo = sys.argv[6]
kernel_fun = sys.argv[7]
if algo == "svc" or algo == "linearsvc": 
	if len(sys.argv) == 9:
		penalty = float(sys.argv[8])
	else:
		penalty = 1.0
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

#linear k fold validation
def kfold_validation(nfold):
	model = build_model()
	#scores = sk.cross_validation.cross_val_score(model, XC, yc, cv=nfold)
	#print scores
	
	offset = 0
	length = dsize / nfold
	for i in range(0, nfold):
		print "....Next fold %d" %(i)
		
		#split data
		(XV,yv,X,y) = split_data(offset, length)
		dvsize = len(XV)

		#train model
		model.fit(X, y) 
		
		#print support vectors
		print_support_vectors(model)
		
		#predict
		print "making predictions..."
		yp = model.predict(XV)

		#show prediction output
		print_prediction_output(dvsize,yv,yp)
		
		offset += length
		
		
		
# random k fold validation
def rfold_validation(nfold):
	max_offset_frac = 1.0 - 1.0 / nfold
	max_offset_frac -= .01
	
	offset = int(dsize * random.random() * max_offset_frac)
	length = dsize / nfold
	print "offset: %d  length: %d" %(offset, length)
	(XV,yv,X,y) = split_data(offset, length)
	dvsize = len(XV)
	
	#build model
	model = build_model()
	
	#train model
	model.fit(X, y) 

	if (not algo == "linearsvc"):
		print "support vectors..." 
		print model.support_vectors_
		print "num of support vectors"
		print model.n_support_

	#predict
	print "making predictions..."
	yp = model.predict(XV)

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
	print "score %f" %(er)
	print "true positive : %.3f" %(float(tp)  / dvsize)
	print "false positive: %.3f" %(float(fp)  / dvsize)
	print "true negative : %.3f" %(float(tn)  / dvsize)
	print "false negative: %.3f" %(float(fn)  / dvsize)

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
		print "showing support vectors..." 
		print model.support_vectors_
		print "num of support vectors"
		print model.n_support_

#prints prediction output
def print_prediction_output(dvsize,yv,yp):
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
	print "error %f" %(er)
	print "true positive : %.3f" %(float(tp)  / dvsize)
	print "false positive: %.3f" %(float(fp)  / dvsize)
	print "true negative : %.3f" %(float(tn)  / dvsize)
	print "false negative: %.3f" %(float(fn)  / dvsize)



if validation == "kfold":
	kfold_validation(5)
else:
	rfold_validation(5)
	
	
	