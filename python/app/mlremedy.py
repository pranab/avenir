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
		
		#get typed fields	
		self.instance = instanceData.split(",")
		self.varFiledIndexes = list()
		for fl in self.costConfig["fields"]:
			if fl["action"] == "change":
				ind = fl["index"]
				dtype = fl["type"]
				self.instance[ind] = getTyped(self.instance[ind], dtype)
				self.varFiledIndexes.append(ind)
		self.numvarFld = len(self.varFiledIndexes)
				
	def isValid(self, args):
		"""
		is the candidate solution valid
		"""
		assertEqual(self.numvarFld, len(args), 
		"no of fields in the candidate don't match with no of variable fields expected {} found {}".format(self.numvarFld, len(args)))
		valid = True
		for i , v in enumerate(args):
			fld = self.costConfig["fields"][i]
			if fl["action"] == "change":
				bval = self.instance[i]
				nval = v
				if (type(fld["cost"]) == list):
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
		for i , v in enumerate(args):
			fld = self.costConfig["fields"][i]
			bval = self.instance[i]
			nval = v
			
			if (type(fld["cost"]) == list):
				for ch in fld["cost"]:
					if ch[0] == bval and ch[1] == nval:
						cost += ch[2]
						break
			else:
				chdir = fl["direction"]
				unit = fld["unit"]
				crate = fld["cost"]
				if chdir == "pos":
					diff = nval - bval
					cost += diff * crate / unit
				elif chdir == "neg":
					diff = bval - nval
					cost += diff * crate / unit

		return cost
	

		
		
		