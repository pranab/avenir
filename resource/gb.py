#!/Users/pranab/Tools/anaconda/bin/python

# Package imports
import os
import sys
sys.path.append(os.path.abspath("../supv"))
from gbt import *

# classifier
gbtClass = GradientBoostedTrees(sys.argv[1])

# override config param
if len(sys.argv) > 2:
	#parameters over riiding config file
	for i in range(2, len(sys.argv)):
		items = sys.argv[i].split("=")
		gbtClass.setConfigParam(items[0], items[1])

# execute		
mode = gbtClass.getMode()
print "running mode: " + mode
if mode == "train":
	gbtClass.train()
elif mode == "trainValidate":
	if gbtClass.getSearchParamStrategy() is None:
		gbtClass.trainValidate()
	else:
		gbtClass.trainValidateSearch()
elif mode == "predict":
	gbtClass.predict()
elif mode == "validate":
	gbtClass.validate()
elif mode == "autoTrain":
	gbtClass.autoTrain()
else:
	print "invalid running mode" 

	
	
	
