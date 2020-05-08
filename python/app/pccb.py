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
		"""
		constructor
		"""
		#members and hourly rates
		self.members = ["KD", "PL", "SP", "DJ",  "DI", "PM"]
		self.memberCost = {"KD" : 55.0, "PL" : 45.0, "SP" : 35.0, "DJ" : 30.0,  "DI" : 40.0, "PM" : 40}
		
		#tasks and percentage participation
		self.taskFront = {"DJ" : 60.0, "SP" : 40}
		self.taskML = {"KD" : 60.0, "PL" : 30.0, "SP" : 10.0}
		self.taskDeploy = {"DI" : 70.0, "PL" : 20.0, "DJ" : 10.0}
		self.taskProjMgmt = {"PM" : 64.0, "KD" : 12.0, "SP" : 12.0, "DI" : 12.0}
		
		#member replacement hourly cost
		self.replCost = 40.0
		
	def taskCost(self, task, hours):
		"""
		cost for a task
		"""
		cost = 0.0
		for k, v in task.items():
			mHours = hours * v / 100.0
			mcost = self.memberCost[k] * mHours
			cost += mcost
		return cost
	
	def intrCost(self, intrCounts, elapsedDays):
		"""
		cost for interruption
		"""
		cost = 0.0
		for i in range(len(intrCounts)):
			cost += 0.25 * elapsedDays * intrCounts[i] * self.memberCost[self.members[i]]
			#print("elapsedDays {}  intr cost {:.3f}".format(elapsedDays, cost))
		return cost


def prCost(args):
	"""
	callback for cost calculation
	"""
	#hours
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
	
	#unexpected
	i = i + 1
	unexpected = args[i]
	i = i + 1
	unexpectedHours = args[i]

	#interruptions
	elapsedHours = 0.8 * (taskFrontHours + taskMLHours + taskDeployHours + taskProjMgmtHours)
	elapsedDays = int(elapsedHours / 8) + 1
	intrCounts = args[i+1:i+7]
	#print(str(intrCounts))


	i = i + 7
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

	tcost += proj.intrCost(intrCounts, elapsedDays)
	
	return tcost
			
if __name__ == "__main__":
	numIter = int(sys.argv[1])
	project = Project()
	simulator = MonteCarloSimulator(numIter, prCost, "./log/mcsim.log", "info")
	
	#task hours
	simulator.registerGaussianSampler(72.0, 8.0)
	simulator.registerNonParametricSampler(40.0, 10.0, 20.0, 28.0, 40.0, 50.0, 60.0, 80.0, 90.0, 100.0, 80.0,\
	65.0, 50.0, 35.0, 32.0, 40.0, 53.0, 70.0, 80.0, 85.0, 90.0, 82.0, 65.0, 45.0, 40.0, 35.0, 30.0, 27.0)
	simulator.registerTriangularSampler(52.0, 78.0, 10.0, 60.0)
	simulator.registerGaussianSampler(80.0, 12.0)
	simulator.registerGaussianSampler(50.0, 5.0)
	
	#unexpected time off or quit
	simulator.registerBernoulliTrialSampler(0.10)
	simulator.registerGaussianSampler(10.0, 2.0)

	#interruptions
	simulator.registerPoissonSampler(5, 8)
	simulator.registerPoissonSampler(4, 6)
	simulator.registerPoissonSampler(2, 3)
	simulator.registerPoissonSampler(6, 9)
	simulator.registerPoissonSampler(3, 5)
	simulator.registerPoissonSampler(2, 4)
	
	simulator.registerExtraArgs(project)
	simulator.run()
	
	print("mean {:.2f}".format(simulator.getMean()))
	print("std dev {:.2f}".format(simulator.getStdDev()))
	critValues = simulator.getUpperTailStat(1.0)
	print("upper critical values")
	for cv in critValues:
		print("{:.3f}  {}".format(cv[0], int(cv[1])))
	simulator.drawHist()
	
	
	

		
