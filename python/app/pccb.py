#!/usr/local/bin/python3

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

# Package imports
import os
import sys
import random
import statistics 
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
from util import *
from sampler import *
from mcsim import *

"""
project cost conf bounds with MC simulation
"""

class Project(object):
	"""
	project plan
	"""
	def __init__(self):
		self.members = {"KD" : 55.0, "PL" : 45.0, "DJ" : 30.0, "SP" : 35.0, "DI" : 40.0, "PM" : 40}
		self.taskFront = {"DJ" : 60.0, "SP" : 40}
		self.taskML = {"KD" : 60.0, "PL" : 30.0, "SP" : 10.0}
		self.taskDeploy = {"DI" : 70.0, "PL" : 20.0, "DJ" : 10.0}
		self.taskProjMgmt = {"PM" : 64.0, "KD" : 12.0, "SP" : 12.0, "DI" : 12.0}
		self.replCost = 40.0
		
	def taskCost(self, task, hours):
		cost = 0.0
		for k, v in task.items():
			mHours = hours * v / 100.0
			mcost = self.members[k] * mHours
			cost += mcost
		return cost

def prCost(args):
	i = 0
	taskFrontHours = args[i]
	i = i + 1
	taskMLHours = args[i]
	i = i + 1
	taskMLLeadParticipation = args[i]
	i = i + 1
	taskDeployHours = args[i]
	i = i + 1
	taskProjMgmtHours = args[i]
	i = i + 1
	unexpected = args[i]
	i = i + 1
	unexpectedHours = args[i]
	i = i + 1
	proj = args[i]
	
	tcost = 0.0
	tcost += proj.taskCost(proj.taskFront, taskFrontHours)
	otherParticipation = 100.0 - taskMLLeadParticipation - 10.0
	taskML = {"KD" : taskMLLeadParticipation, "PL" : otherParticipation, "SP" : 10.0}
	tcost += proj.taskCost(taskML, taskMLHours)
	tcost += proj.taskCost(proj.taskDeploy, taskDeployHours) 
	tcost += proj.taskCost(proj.taskProjMgmt, taskProjMgmtHours) 
	
	if unexpected:
		tcost += unexpectedHours * proj.replCost
	
	return tcost
			
if __name__ == "__main__":
	numIter = int(sys.argv[1])
	project = Project()
	simulator = MonteCarloSimulator(numIter, prCost)
	
	#task hours
	simulator.registerGaussianSampler(72.0, 8.0)
	simulator.registerNonParametricSampler(40.0, 10.0, 20.0, 28.0, 40.0, 50.0, 60.0, 80.0, 75.0,\
	65.0, 50.0, 35.0, 32.0, 40.0, 53.0, 70.0, 80.0, 85.0, 90.0, 82.0, 65.0, 45.0, 40.0, 35.0, 30.0, 27.0)
	simulator.registerTriangularSampler(52.0, 78.0, 10.0, 60.0)
	simulator.registerGaussianSampler(80.0, 12.0)
	simulator.registerGaussianSampler(50.0, 5.0)
	
	#unexpected time off or quit
	simulator.registerBinomialSampler(0.10)
	simulator.registerGaussianSampler(10.0, 2.0)
	
	simulator.registerExtraArgs(project)
	simulator.run()
	critValues = simulator.getUpperTailStat(1.0)
	print("upper critical values")
	for cv in critValues:
		print("{:.3f}  {}".format(cv[0], int(cv[1])))
	simulator.drawHist()
	
	
	

		
