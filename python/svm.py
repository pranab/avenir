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
	print "usage: <training_data_file> <index_of_feature_fields> <index_of_class_field> <preprocess> <algorithm> <kernel> <penalty>"
	sys.exit()

data_file = sys.argv[1]
feat_field_indices = sys.argv[2].split(",")
feat_field_indices = [int(a) for a in feat_field_indices]
class_field_index = int(sys.argv[3])
preprocess = sys.argv[4]
algo = sys.argv[5]
kernel_fun = sys.argv[6]
if algo == "svc" or algo == "linearsvc": 
	if len(sys.argv) == 8:
		penalty = float(sys.argv[7])
	else:
		penalty = 1.0
print feat_field_indices


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

yc = d[:,[class_field_index]]
yc = yc.reshape(dsize)
yc = [int(a) for a in yc]

print XC
print yc

# validation set
vlo = int(dsize * random.random() * 0.5)
vup = vlo + dsize / 5
XV = XC[vlo:vup:1]
yv = yc[vlo:vup:1]
dvsize = len(XV)
print "data size %d validation data size %d" %(dsize, dvsize)
#print "validation set"
#print XV
#print yv

#training set
X = np.delete(XC, np.s_[vlo:vup:1], 0)
y = np.delete(yc, np.s_[vlo:vup:1], 0)
#print "training set"
#print X
#print y

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
	
model.fit(X, y) 

if (not algo == "linearsvc"):
	print "support vectors..." 
	print model.support_vectors_
	print "num of support vectors"
	print model.n_support_

yp = model.predict(XV)

print "making predictions..."
err_count = 0
for r in range(0,dvsize):
	print "actual: %d  predicted: %d" %(yv[r], yp[r])
	if (not yv[r] ==  yp[r]):
		err_count += 1
		
er = float(err_count) * 100.0 / dvsize		
print "error percentage %f" %(er)
	
	
	
	
	