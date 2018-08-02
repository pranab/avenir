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
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tag import StanfordNERTagger
from collections import defaultdict
import pickle
sys.path.append(os.path.abspath("../lib"))
from util import *

#text preprocessor
class TextPreProcessor:
	def __init__(self, stemmer = "lancaster", verbose=False):
		self.verbose = verbose
		self.lemmatizer = WordNetLemmatizer()

	def stripHtml(self, text):
		soup = BeautifulSoup(text, "html.parser")
		return soup.get_text()

	def removeBetweenSquareBrackets(self, text):
		return re.sub('\[[^]]*\]', '', text)

	def denoiseText(self, text):
		text = stripHtml(text)
 		text = removeBetweenSquareBrackets(text)
 		return text

	def replaceContractions(self, text):
		"""Replace contractions in string of text"""
		return contractions.fix(text)

	def tokenize(self, text):
		words = nltk.word_tokenize(text)
		return words

	def removeNonAscii(self, words):
		"""Remove non-ASCII characters from list of tokenized words"""
		newWords = []
		for word in words:
			newWord = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
			newWords.append(newWord)
		return newWords

	def allow(self, words):
		""" allow only specific charaters """
		allowed = [word for word in words if re.match('^[A-Za-z0-9\.\,\:\;\!\?\(\)\'\-\$\@\%\"]+$', word) is not None]		
		return allowed		
		
	def toLowercase(self, words):
		"""Convert all characters to lowercase from list of tokenized words"""
		newWords = [word.lower() for word in words]
		return newWords

	def removePunctuation(self, words):
		"""Remove punctuation from list of tokenized words"""
		newWords = []
		for word in words:
			newWord = re.sub(r'[^\w\s]', '', word)
			if newWord != '':
				newWords.append(newWord)
		return newWords

	def replaceNumbers(self, words):
		"""Replace all interger occurrences in list of tokenized words with textual representation"""
		p = inflect.engine()
		newWords = []
		for word in words:
			if word.isdigit():
				newWord = p.number_to_words(word)
				newWords.append(newWord)
			else:
				newWords.append(word)
		return newWords

	def removeStopwords(self, words):
		"""Remove stop words from list of tokenized words"""
		newWords = []
		for word in words:
			if word not in stopwords.words('english'):
				newWords.append(word)
		return newWords

	def removeCustomStopwords(self, words, stopWords):
		"""Remove stop words from list of tokenized words"""
		removed = [word for word in words if word not in stopWords]		
		return removed

	def removeLowFreqWords(self, words, minFreq):
		"""Remove low frewquncy words from list of tokenized words"""
		frequency = defaultdict(int)
		for word in words:
			frequency[word] += 1
		removed = [word for word in words if frequency[word] > minFreq]		
		return removed	

	def removeNumbers(self, words):
		"""Remove numbers"""
		removed = [word for word in words if not isNumber(word)]		
		return removed		

	def removeShortWords(self, words, minLengh):
		"""Remove short words """
		removed = [word for word in words if len(word) >= minLengh]		
		return removed		

	def keepAllowedWords(self, words, keepWords):
		"""Keep  words from the list only"""
		kept = [word for word in words if word in keepWords]		
		return kept

	def stemWords(self, words):
		"""Stem words in list of tokenized words"""
		if stemmer == "lancaster":
			stemmer = LancasterStemmer()
		elif stemmer == "snowbal":
			stemmer = SnowballStemmer()
		elif stemmer == "porter":
			stemmer = PorterStemmer()
		stems = [stemmer.stem(word) for word in words]
		return stems

	def lemmatizeWords(self, words):
		"""Lemmatize tokens in list of tokenized words"""
		lemmas = [self.lemmatizer.lemmatize(word) for word in words]
		return lemmas

	def lemmatizeVerbs(self, words):
		"""Lemmatize verbs in list of tokenized words"""
		lemmas = [self.lemmatizer.lemmatize(word, pos='v') for word in words]
		return lemmas

	def normalize(self, words):
		words = self.removeNonAscii(words)
		words = self.toLowercase(words)
		words = self.removePunctuation(words)
		words = self.replaceNumbers(words)
		words = self.removeStopwords(words)
		return words

	def posTag(self, textTokens):
		tags = nltk.pos_tag(textTokens)
		return tags

	def extractEntity(self, textTokens, classifierPath, jarPath):
		st = StanfordNERTagger(classifierPath, jarPath) 
		entities = st.tag(textTokens)
		return entities

	def documentFeatures(self, document, wordFeatures):
		documentWords = set(document)
		features = {}
		for word in wordFeatures:
			features[word] = (word in documentWords)
		return features

# TF IDF 
class TfIdf:
	# initialize
	def __init__(self, vocFilt, doIdf):
		self.vocFilt = vocFilt
		self.doIdf = doIdf
		self.wordCounter = {}
		self.wordFreq = {}
		self.wordInDocCount = {}
		self.docCount = 0
		self.corpSize = 0
		self.freqDone = False
		self.vocabulary = set()
		self.wordIndex = None
	
	# count words in a doc
	def countDocWords(self, words):
		print "doc size " + str(len(words))
		for word in words:
			if self.vocFilt is None or word in self.vocFilt:
				count = self.wordCounter.get(word, 0)
				self.wordCounter[word] = count + 1
		self.corpSize += len(words)
		self.vocabulary.update(words)

		if (self.doIdf):
			self.docCount += 1
			for word in set(words):
				self.wordInDocCount.get(word, 0)
				self.wordInDocCount[word] = count + 1
		self.freqDone = False
	
	# get tfidf for corpus
	def getWordFreq(self):
		print "counter size " + str(len(self.wordCounter))
		if not self.freqDone:
			for item in self.wordCounter.items():
				self.wordFreq[item[0]] = float(item[1]) / self.corpSize					
			if self.doIdf:
				for k in self.wordFreq.keys():
					self.wordFreq.items[k] *=  math.log(self.docCount / self.wordInDocCount.items[k])	
			self.freqDone = True
		return self.wordFreq
	
	# reset counter
	def resetCounter(self):
		self.wordCounter = {}

	# build vocbulary
	def buildVocabulary(self, words):
		self.vocabulary.update(words)

	# return vocabulary
	def getVocabulary(self):
		return self.vocabulary
	
	# index for all words in vcabulary
	def creatWordIndex(self):
		self.wordIndex = {word : idx for idx, word in enumerate(list(self.vocabulary))}

	# save 
	def save(self, saveFile):
		sf = open(saveFile, "wb")
		pickle.dump(self, sf)
		sf.close()

	# load 
	@staticmethod
	def load(saveFile):
		sf = open(saveFile, "rb")
		tfidf = pickle.load(sf)
		sf.close()
		return tfidf

# clean doc to create term array
def clean(doc, preprocessor, verbose):
	if verbose:
		print "--raw doc"
		print doc
	words = preprocessor.tokenize(doc)
	words = preprocessor.allow(words)
	words = preprocessor.toLowercase(words)
	words = preprocessor.removeStopwords(words)
	words = preprocessor.removeShortWords(words, 3)
	words = preprocessor.removePunctuation(words)
	words = preprocessor.lemmatizeWords(words)
	if verbose:
		print "--after pre processing"
		print words
	return words


