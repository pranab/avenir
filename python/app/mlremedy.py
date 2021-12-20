#!/usr/local/bin/python3

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
import jprops
import numpy as np
import statistics 
import json
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
from util import *
from optpopu import *
from optsolo import *
from sampler import *

"""
Remedial action or prescriptive analytic with counter factuals and machine learning models
"""

class RemedyCost(object):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, costConfigFile, mlConfigFile, modelType, instanceData):
		"""
		intialize
		"""
		self.logger = None
		if modelType == "ffnn":
			self.model = FeedForwardNetwork(mlConfigFile)
		else:
			exitWithMsg("invalid ml model type")

		#cost config
		with open(costConfigFile, 'r') as contentFile:
			self.costConfig = json.load(contentFile)
		
		types = "0:string,1:cat:M F,2:int,3:int,4:int,5:int,6:cat:NS SS SM,7:cat:BA AV GO,8:int,9:int,10:cat:WH BL SA EA,11:int"
		tdata = getFileAsTypedRecords(filePath, types)


		self.instance = instanceData.split(",")
		
		#make all fields typed
		typeSpec = ""
		self.varFiledIndexes = list()
		for i, fl in enumerate(self.costConfig["fields"]):
			sp = str(i) + ":" + fl["type"]
			typeSpec += sp
			if fl["action"] == "change":
				self.varFiledIndexes.append(i)
				
		typeSpec = typeSpec[:-1]
		self.instance = getRecAsTypedRecord(self.instance, typeSpec, ",")
		self.numvarFld = len(self.varFiledIndexes)
				
	def isValid(self, args):
		"""
		is the candidate solution valid
		"""
		assertEqual(self.numvarFld, len(args), 
		"no of fields in the candidate don't match with no of variable fields expected {} found {}".format(self.numvarFld, len(args)))
		valid = True
		minst = self.__getMutatedInstance(args)
		for i , vfi in enumerate(self.varFiledIndexes):
			fl = self.costConfig["fields"][vfi]
			if fl["action"] == "change":
				bval = self.instance[vfi]
				nval = args[i]
				if (type(fl["cost"]) == list):
					valid = False
					for ch in fld["cost"]:
						if ch[0] == bval and ch[1] == nval:
							valid = True
							break
					if not valid:
						break
				else:
					chdir = fl["direction"]
					if chdir == "pos" and nval <= bval:
						valid = False
						break
					elif chdir == "neg" and nval >= bval:
						valid = False
						break
			else:
				exitWithMsg("field action is not change")
				
		return valid
		
	def evaluate(self, args):
		"""
		get cost
		"""
		cost = 0.0
		minst = self.__getMutatedInstance(args)
		for i , vfi in enumerate(self.varFiledIndexes):
			fl = self.costConfig["fields"][vfi]
			bval = self.instance[vfi]
			nval = args[i]
			
			if (type(fld["cost"]) == list):
				for ch in fld["cost"]:
					if ch[0] == bval and ch[1] == nval:
						cost += ch[2]
						break
			else:
				chdir = fl["direction"]
				unit = fl["unit"]
				crate = fl["cost"]
				if chdir == "pos":
					diff = nval - bval
					cost += diff * crate / unit
				elif chdir == "neg":
					diff = bval - nval
					cost += diff * crate / unit

		return cost
	
	def printSoln(self, cand):
		"""
		
		"""
		fsoln = __getMutatedInstance(self, cand.soln)
		print(fsoln)
		
	
	def __getMutatedInstance(self, args):
		"""
		create instance with mutated field string values
		"""
		instance = self.instance.copy()
		i = 0
		for fl in self.costConfig["fields"]:
			if fl["action"] == "change":
				ind = fl["index"]
				instance[ind] = str(arg[i])
				i += 1
		return instance
		

if __name__ == "__main__":
	assert len(sys.argv) == 6, "wrong command line args"
	optConfFile = sys.argv[1]
	mlConfFile = sys.argv[2]
	modelType  = sys.argv[3]
	instanceData =  sys.argv[4]
	costConfFile = sys.argv[5]
	
	#create optimizer
	remCost = RemedyCost(costConfFile, mlConfFile, modelType, instanceData)
	optimizer = GeneticAlgorithmOptimizer(optConfFile, remCost)
	remCost.logger = optimizer.logger	
	config = optimizer.getConfig()
	
	#run optimizer
	optimizer.run()
	print("optimizer started, check log file for output details...")
	
	#best soln
	print("\nbest solution found")
	best = optimizer.getBest()
	remCost.pritnSoln(best)

	#output soln tracker
	if optimizer.trackingOn:
		print("\nbest solution history")
		print(str(optimizer.tracker))
	
	#local search	
	locBest = optimizer.getLocBest()
	if locBest is not None:
		print("\nbest solution after local search of global best solution")
		pritnSoln(locBest, schedCost)
	
		if (locBest.cost < best.cost):
			print("\nlocally search solution is best overall")
		else:
			print("\nlocal search failed to find a better solution")
		
	print("\ntotal solution count {}  invalid solution count {}".format(schedCost.solnCount, schedCost.invalidSonlCount))
	
	
			