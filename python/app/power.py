#!/usr/bin/python

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

import os
import sys
from random import randint
import time
from datetime import datetime
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *
from sampler import *



if __name__ == "__main__":
	op = sys.argv[1]
	
	if op == "gen":
		startNumDaysPast = int(sys.argv[2])
		curTime = int(time.time())
		startTime = curTime - (startNumDaysPast  + 1) * secInDay
		startTime = (startTime / secInHour) * secInHour
		intv = 60 * 60
		sampTime = startTime
		
		mean = 3.0
		trend = 0.5 / (365 * 24)
		yearCycle = [0.75, 0.48, 0.22, -0.6, -0.08, 0.19, 0.40, 0.68, 0.41, 0.12, 0.39, .72]
		dayCycle = [-0.10, -0.12, -0.16, -0.24, -0.28, -0.13, -0.08, 0.12, 0.25, 0.37, 0.45,\
			0.53, 0.42, 0.34, 0.26, 0.21, 0.16, 0.12, 0.10, 0.06, -0.01, -0.05, -0.08, -0.10]
		noiseDistr = GaussianRejectSampler(0, .05)


		cureTrend = 0
		while (sampTime < curTime):
			usage = mean + cureTrend 

			secIntoYear = sampTime % secInYear
			month = int(secIntoYear / secInMonth)
			#print("month {}".format(month))
			usage += yearCycle[month]

			hourIntoDay = int((sampTime % secInDay) / secInHour)
			#print("hour {}".format(hourIntoDay))
			usage += dayCycle[hourIntoDay]

			usage += noiseDistr.sample()

			dt = datetime.fromtimestamp(sampTime)
			dt = dt.strftime("%Y-%m-%d %H:%M:%S")
			print ("{},{:.3f}".format(dt, usage))

			cureTrend += trend
			sampTime += secInHour



