#!/Users/pranab/Tools/anaconda/bin/python

# Package imports
import os
import sys
sys.path.append(os.path.abspath("../supv"))
from rf import *

# classifier
rfClass = RandomForest(sys.argv[1])

# override config param
if len(sys.argv) > 2:
	#parameters over riiding config file
	for i in range(2, len(sys.argv)):
		items = sys.argv[i].split("=")
		rfClass.setConfigParam(items[0], items[1])

# execute		
mode = rfClass.getMode()
print "running mode: " + mode
if mode == "train":
	rfClass.train()
elif mode == "trainValidate":
	if rfClass.getSearchParamStrategy() is None:
		rfClass.trainValidate()
	else:
		rfClass.trainValidateSearch()
elif mode == "predict":
	clsData = rfClass.predict()
	print clsData
elif mode == "validate":
	rfClass.validate()
elif mode == "autoTrain":
	rfClass.autoTrain()
else:
	print "invalid running mode" 

	
	
	
