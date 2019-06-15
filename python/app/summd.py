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

# Package imports
import os
import sys
import jprops
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from preprocess import *
from summ import *

def display(config, sumSenteces):
	#print "num sentences " + str(len(sumSenteces)) 
	showScore = config.getBooleanConfig("common.show.score")[0]
	for sen in sumSenteces:
		sent =  sen[0]
		if showScore:
			sent = sent + "  (" + str(sen[1]) + ")"
		print sent
		
if __name__ == "__main__":
	configFile  = sys.argv[1]
	op  = sys.argv[2]
	if op == "tfSumm":
		summarizer = TermFreqSumm(configFile)
		config = summarizer.getConfig()
		verbose = config.getBooleanConfig("common.verbose")[0]
		filePath = config.getStringConfig("common.data.file")[0]
		sumSenteces = summarizer.getSummary(filePath)
		display(config, sumSenteces)
	elif op == "sbSumm":
		summarizer = SumBasicSumm(configFile)
		config = summarizer.getConfig()
		verbose = config.getBooleanConfig("common.verbose")[0]
		filePath = config.getStringConfig("common.data.file")[0]
		sumSenteces = summarizer.getSummary(filePath)
		display(config, sumSenteces)
	elif op == "lsSumm":
		summarizer = LatentSemSumm(configFile)
		config = summarizer.getConfig()
		verbose = config.getBooleanConfig("common.verbose")[0]
		filePath = config.getStringConfig("common.data.file")[0]
		sumSenteces = summarizer.getSummary(filePath)
		display(config, sumSenteces)
	elif op == "nmfSumm":
		summarizer = NonNegMatFactSumm(configFile)
		config = summarizer.getConfig()
		verbose = config.getBooleanConfig("common.verbose")[0]
		filePath = config.getStringConfig("common.data.file")[0]
		sumSenteces = summarizer.getSummary(filePath)
		display(config, sumSenteces)
	elif op == "trSumm":
		summarizer = TextRankSumm(configFile)
		config = summarizer.getConfig()
		verbose = config.getBooleanConfig("common.verbose")[0]
		filePath = config.getStringConfig("common.data.file")[0]
		sumSenteces = summarizer.getSummary(filePath)
		display(config, sumSenteces)
	elif op == "etrSumm":
		summarizer = EmbeddingTextRankSumm(configFile)
		config = summarizer.getConfig()
		verbose = config.getBooleanConfig("common.verbose")[0]
		filePath = config.getStringConfig("common.data.file")[0]
		sumSenteces = summarizer.getSummary(filePath)
		display(config, sumSenteces)
	
