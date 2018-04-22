#!/Users/pranab/Tools/anaconda/bin/python

# Package imports
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import matplotlib
import random
import jprops

class BaseClassifier:
	def __init__(self):
		pass
		
	def autoTrain(self):
		trainErr = self.train()
		
