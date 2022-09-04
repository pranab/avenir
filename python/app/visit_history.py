#!/usr/local/bin/python3

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

import os
import sys
from random import randint
import time
import uuid
import argparse
sys.path.append(os.path.abspath("../supv"))
from matumizi.util import *
from mcclf import *

def  genVisitHistory(numUsers, convRate, label):
	for i in range(numUsers):
		userID = genID(12)
		userSess = []
		userSess.append(userID)

		conv = randint(0, 100)
		if (conv < convRate):
			#converted
			if (label):
				if (randint(0,100) < 90):
					userSess.append("T")
				else:
					userSess.append("F")
				
				
			numSession = randint(2, 20)
			for j in range(numSession):
				sess = randint(0, 100)
				if (sess <= 15):
					elapsed = "H"
				elif (sess > 15 and sess <= 40):
					elapsed = "M"
				else:
					elapsed = "L"

				sess = randint(0, 100)
				if (sess <= 15):
					duration = "L"
				elif (sess > 15 and sess <= 40):
					duration = "M"
				else:
					duration = "H"
					
				sessSummary = elapsed + duration
				userSess.append(sessSummary)
				
				
		else:
			#not converted
			if (label):
				if (randint(0,100) < 90):
					userSess.append("F")
				else:
					userSess.append("T")

			numSession = randint(2, 12)
			for j in range(numSession):
				sess = randint(0, 100)
				if (sess <= 20):
					elapsed = "L"
				elif (sess > 20 and sess <= 45):
					elapsed = "M"
				else:
					elapsed = "H"

				sess = randint(0, 100)
				if (sess <= 20):
					duration = "H"
				elif (sess > 20 and sess <= 45):
					duration = "M"
				else:
					duration = "L"
								 	
				sessSummary = elapsed + duration
				userSess.append(sessSummary)

		print(",".join(userSess))
				 
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--nuser', type=int, default = 100, help = "num of users")
	parser.add_argument('--crate', type=int, default = 10, help = "concersion rate")
	parser.add_argument('--label', type=str, default = "false", help = "whether to add label")
	parser.add_argument('--mlfpath', type=str, default = "false", help = "ml config file")
	args = parser.parse_args()
	op = args.op
	
	if op == "gen":	
		numUsers = args.nuser
		convRate = args.crate
		
		label = args.label == "true"
		genVisitHistory(numUsers, convRate, label)
		
	elif op == "train":
		model = MarkovChainClassifier(args.mlfpath)
		model.train()
		
	elif op == "pred":
		model = MarkovChainClassifier(args.mlfpath)
		model.predict()

	else:
		exitWithMsg("invalid command)")