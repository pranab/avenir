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
import numpy
import statistics 
import math
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from util import *
from mlutil import *
from txproc import *
from nlm import *


class SegmentSearch(object):
	def __init__(self, passageSize = 4):
		"""
		initialize
		"""
		self.minParNl = 2
		fr = TextFragmentGenerator("passage", self.minParNl, passageSize, False)
		self.matcher = SemanticSimilaityBiEnc(fr, False)

	def passageSearch(self, query, docs):
		"""
		passage search for long text query
		"""
		self.matcher.clear()
		self.matcher.addNamedText(docs)

		qparas = getParas(query, self.minParNl)
		for qp in qparas:
			self.matcher.search(qp, "tsavm")	
	
		res = self.matcher.getFragDocMatches()
		return res
		
	def getPassages(self, dname):
		"""
		get matched passages from a document
		"""
		return self.matcher.getMatchedSegments(dname)
