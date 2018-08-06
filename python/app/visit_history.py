#!/usr/bin/python

import os
import sys
from random import randint
import time
import uuid
sys.path.append(os.path.abspath("../lib"))
from util import *


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

		print ",".join(userSess)
				 

numUsers = int(sys.argv[1])
convRate = int(sys.argv[2])
label = False
if (len(sys.argv) == 4):
	label = sys.argv[3] == "label"
genVisitHistory(numUsers, convRate, label)

	