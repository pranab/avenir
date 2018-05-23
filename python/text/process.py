#!/usr/bin/python

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





