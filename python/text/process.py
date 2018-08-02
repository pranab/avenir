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
import random
import time
from datetime import datetime
import re, string, unicodedata
import nltk
import contractions
import inflect
from nltk.tag import StanfordNERTagger
from collections import defaultdict
import gensim

#text preprocessor
class TextProcessor:
	classifier = None
	ldaModel = None

	def __init__(self, verbose=False):
		self.verbose = verbose

	def posTag(self, textTokens):
		""" extract POS tags """
		tags = nltk.pos_tag(textTokens)
		return tags

	def extractEntity(self, textTokens, classifierPath, jarPath):
		""" extract entity """
		st = StanfordNERTagger(classifierPath, jarPath) 
		entities = st.tag(textTokens)
		return entities
	
	def trainNaiveBayesClassifier(self, trainSet, testSet):
		"""  train naive bayes classifier  """
		self.classifier = nltk.NaiveBayesClassifier.train(trainSet)
		acc = nltk.classify.accuracy(self.classifier, testSet)
		return (classifier, acc)
	
	def trainMaxEntClassifier(self, trainSet, testSet):
		"""  train max entropy classifier  """
		self.classifier = nltk.MaxentClassifier.train(trainSet, "megam")
		acc = nltk.classify.accuracy(self.classifier, testSet)
		return (classifier, acc)

	def predict(self, testSet):
		""" predict """
		pred = self.classifier.classify(testSet)
		return pred
	
	def ldaModel(self, docTermMatrix, numTopics, dictionary, numPasses = 50):
		"""  LDA for topic modeling  """
		self.ldaModel = gensim.models.ldamodel.LdaModel(docTermMatrix, num_topics=numTopics, id2word=dictionary, passes=numPasses)
		return self.ldaModel





