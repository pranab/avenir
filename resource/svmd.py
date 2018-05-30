#!/usr/bin/python

# Package imports
import os
import sys
sys.path.append(os.path.abspath("../supv"))
from svm import *

# classifier
svmClass = SupportVectorMachine(sys.argv[1])

# override config param
if len(sys.argv) > 2:
	#parameters over riiding config file
	for i in range(2, len(sys.argv)):
		items = sys.argv[i].split("=")
		svmClass.setConfigParam(items[0], items[1])

# execute		
mode = svmClass.getMode()
print "running mode: " + mode
if mode == "train":
	svmClass.train()
elif mode == "trainValidate":
	if svmClass.getSearchParamStrategy() is None:
		svmClass.trainValidate()
	else:
		svmClass.trainValidateSearch()
elif mode == "predict":
	svmClass.predict()
elif mode == "validate":
	svmClass.validate()
elif mode == "autoTrain":
	svmClass.autoTrain()
else:
	print "invalid running mode" 

