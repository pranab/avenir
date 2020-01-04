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
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import pickle
import numpy as np
sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *

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
			if isinstance(word, unicode):
				newWord = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore')
			else:
				newWord = word
			newWords.append(newWord)
		return newWords

	def replaceNonAsciiFromText(self, text):
		""" replaces non ascii with blank  """
		return ''.join([i if ord(i) < 128 else ' ' for i in text])

	def removeNonAsciiFromText(self, text):
		""" replaces non ascii with blank  """
		return ''.join([i if ord(i) < 128 else '' for i in text])
	
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
#NGram
class NGram:
	def __init__(self, vocFilt, verbose=False):
		"""
		initialize
		"""
		self.vocFilt = vocFilt
		self.nGramCounter = dict()
		self.nGramFreq = dict()
		self.corpSize = 0
		self.vocabulary = set()
		self.freqDone = False
		self.verbose = verbose
		self.vecWords = None
		self.nonZeroCount = 0
		
	def countDocNGrams(self, words):
		"""
		count words in a doc
		"""
		if self.verbose:
			print ("doc size " + str(len(words)))
		nGrams = self.toNGram(words)
		for nGram in nGrams:
			count = self.nGramCounter.get(nGram, 0)
			self.nGramCounter[nGram] = count + 1
			self.corpSize += 1
		self.vocabulary.update(words)	

	def remLowCount(self, minCount):
		"""
		removes items with count below threshold
		"""
		self.nGramCounter = dict(filter(lambda item: item[1] >= minCount, self.nGramCounter.items()))
		
	def getVocabSize(self):
		"""
		get vocabulary size
		"""
		return len(self.nGramCounter)
		
	def getNGramFreq(self):
		"""
		get normalized count
		"""
		if self.verbose:
			print ("counter size " + str(len(self.nGramCounter)))
		if not self.freqDone:
			for item in self.nGramCounter.items():
				self.nGramFreq[item[0]] = float(item[1]) / self.corpSize					
			self.freqDone = True
		return self.nGramFreq
	
	def getNGramIndex(self, show):
		"""
		convert to list
		"""
		if self.vecWords is None:
			self.vecWords = list(self.nGramCounter)
			if show:
				for vw in enumerate(self.vecWords):
					print(vw)
			
	def getVector(self, words, byCount, normalized):
		"""
		convert to vector
		"""
		if self.vecWords is None:
			self.vecWords = list(self.nGramCounter)
		
		nGrams = self.toNGram(words)
		if self.verbose:
			print("vocabulary size {}".format(len(self.vecWords)))
			print("ngrams")
			print(nGrams)
		self.nonZeroCount = 0
		vec = list(map(lambda vw: self.getVecElem(vw, nGrams, byCount, normalized), self.vecWords))
		return vec
	
	def getVecElem(self, vw, nGrams, byCount, normalized):
		"""
		get vector element
		"""
		if vw in nGrams:
			if byCount:
				if normalized:
					el = self.nGramFreq[vw]
				else:
					el = self.nGramCounter[vw]
			else:
				el = 1
			self.nonZeroCount += 1
		else:
			if (byCount and normalized):
				el = 0.0
			else:
				el = 0
		return el
	
	def getNonZeroCount(self):
		"""
		get non zero vector element count
		"""
		return self.nonZeroCount
		
	def toBiGram(self, words):
		"""
		convert to bigram
		"""
		if self.verbose:
			print ("doc size " + str(len(words)))
		biGrams = list()
		for i in range(len(words)-1):
			w1 = words[i]
			w2 = words[i+1]
			if self.vocFilt is None or (w1 in self.vocFilt and w2 in self.vocFilt):
				nGram = (w1, w2)
				biGrams.append(nGram)
		return biGrams

	def toTriGram(self, words):
		"""
		convert to trigram
		"""
		if self.verbose:
			print ("doc size " + str(len(words)))
		triGrams = list()
		for i in range(len(words)-2):
			w1 = words[i]
			w2 = words[i+1]
			w3 = words[i+2]
			if self.vocFilt is None or (w1 in self.vocFilt and w2 in self.vocFilt and w3 in self.vocFilt):
				nGram = (w1, w2, w3)
				triGrams.append(nGram)
		return triGrams

	def save(self, saveFile):
		"""
		save 
		"""
		sf = open(saveFile, "wb")
		pickle.dump(self, sf)
		sf.close()

	@staticmethod
	def load(saveFile):
		"""
		load
		"""
		sf = open(saveFile, "rb")
		nGrams = pickle.load(sf)
		sf.close()
		return nGrams
		
# TF IDF 
class TfIdf:
	def __init__(self, vocFilt, doIdf, verbose=False):
		"""
		initialize
		"""
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
		self.verbose = verbose
		self.vecWords = None
	
	def countDocWords(self, words):
		"""
		count words in a doc
		"""
		if self.verbose:
			print ("doc size " + str(len(words)))
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
	
	
	def getWordFreq(self):
		"""
		get tfidf for corpus
		"""
		if self.verbose:
			print ("counter size " + str(len(self.wordCounter)))
		if not self.freqDone:
			for item in self.wordCounter.items():
				self.wordFreq[item[0]] = float(item[1]) / self.corpSize					
			if self.doIdf:
				for k in self.wordFreq.keys():
					self.wordFreq.items[k] *=  math.log(self.docCount / self.wordInDocCount.items[k])	
			self.freqDone = True
		return self.wordFreq
	
	def getCount(self, word):
		"""
		get counter
		"""
		if word in self.wordCounter:
			count = self.wordCounter[word]
		else:
			raise ValueError("word not found in count table " + word)
		return count
		
	def getFreq(self, word):
		"""
		get normalized frequency
		"""
		if word in self.wordFreq:
			freq = self.wordFreq[word]
		else:
			raise ValueError("word not found in count table " + word)
		return freq

	def resetCounter(self):
		"""
		reset counter
		"""
		self.wordCounter = {}

	def buildVocabulary(self, words):
		"""
		build vocbulary
		"""
		self.vocabulary.update(words)

	def getVocabulary(self):
		"""
		return vocabulary
		"""
		return self.vocabulary
	
	def creatWordIndex(self):
		"""
		index for all words in vcabulary
		"""
		self.wordIndex = {word : idx for idx, word in enumerate(list(self.vocabulary))}

	def getVector(self, words, byCount, normalized):
		"""
		get vector
		"""
		if self.vecWords is None:
			self.vecWords = list(self.wordCounter)
		vec = list(map(lambda vw: self.getVecElem(vw, words, byCount, normalized), self.vecWords))
		return vec
	
	def getVecElem(self, vw, words, byCount, normalized):
		"""
		vector element
		"""
		el = 0
		if vw in words:
			if byCount:
				if normalized:
					el = self.wordFreq[vw]
				else:
					el = self.wordCounter[vw]
			else:
				el = 1
		return el
				
	def save(self, saveFile):
		"""
		save
		"""
		sf = open(saveFile, "wb")
		pickle.dump(self, sf)
		sf.close()

	# load 
	@staticmethod
	def load(saveFile):
		"""
		load
		"""
		sf = open(saveFile, "rb")
		tfidf = pickle.load(sf)
		sf.close()
		return tfidf

# bigram
class BiGram(NGram):
	def __init__(self, vocFilt, verbose=False):
		"""
		initialize
		"""
		super(BiGram, self).__init__(vocFilt, verbose)

	def toNGram(self, words):
		"""
		convert to Ngrams
		"""
		return self.toBiGram(words)

# trigram
class TriGram(NGram):
	def __init__(self, vocFilt, verbose=False):
		"""
		initialize
		"""
		super(TriGram, self).__init__(vocFilt, verbose)

	def toNGram(self, words):
		"""
		convert to Ngrams
		"""
		return self.toTriGram(words)
	

# sentence processor
class DocSentences:
	# initialize
	def __init__(self, filePath, minLength, verbose, text=None):
		if filePath:
			self.filePath = filePath
			with open(filePath, 'r') as contentFile:
				content = contentFile.read()
		elif text:
			content = text
		else:
			raise valueError("either file path or text must be provided")

		#self.sentences = content.split('.')
		self.verbose = verbose
		tp = TextPreProcessor()
		content = tp.removeNonAsciiFromText(content)
		sentences = sent_tokenize(content)
		self.sentences = list(filter(lambda s: len(nltk.word_tokenize(s)) >= minLength, sentences))
		if self.verbose:
			print ("num of senteces after length filter " + str(len(self.sentences)))
		self.sentencesAsTokens = [clean(s, tp, verbose) for s in self.sentences]	
	
	# get sentence tokens
	def getSentencesAsTokens(self):
		return self.sentencesAsTokens
	
	# get sentences
	def getSentences(self):
		return self.sentences
	
	# build term freq table
	def getTermFreqTable(self):
		# term count table for all words
		termTable = TfIdf(None, False)
		sentWords = self.getSentencesAsTokens()
		for seWords in sentWords:
			termTable.countDocWords(seWords)
		return termTable

# sentence processor
class WordVectorContainer:
	def __init__(self, dirPath, verbose):
		"""
		initialize
		"""
		self.docs = list()
		self.wordVectors = list()
		self.tp = TextPreProcessor()
		self.similarityAlgo = "cosine"
		self.simAlgoNormalizer = None
		self.termTable = None


	def addDir(self, dirPath):
		"""
		add content of all files ina directory
		"""
		docs, filePaths  = getFileContent(dirPath, verbose)
		self.docs.extend(docs)
		self.wordVectors.extend([clean(doc, self.tp, verbose) for doc in docs])
	
	def addFile(self, filePath):
		"""
		add file content
		"""
		with open(filePath, 'r') as contentFile:
			content = contentFile.read()
		self.wordVectors.append(clean(content, self.tp, verbose))
	
	def addText(self, text):
		"""
		add text
		"""
		self.wordVectors.append(clean(text, self.tp, verbose))

	def addWords(self, words):
		"""
		add words
		"""
		self.wordVectors.append(words)

	def withSimilarityAlgo(self, algo, normalizer=None):
		"""
		set similarity algo
		"""
		self.similarityAlgo = algo
		self.simAlgoNormalizer = normalizer
		
	def getDocsWords(self):
		"""
		get word vectors
		"""
		return self.wordVectors

	def getDocs(self):
		"""
		get docs
		"""
		return self.docs
	
	def getTermFreqTable(self):
		"""
		term count table for all words
		"""
		self.termTable = TfIdf(None, False)
		for words in self.wordVectors:
			self.termTable.countDocWords(words)
		self.termTable.getWordFreq()
		return self.termTable

	def getPairWiseSimilarity(self, byCount, normalized):
		"""
		pair wise similarity
		"""
		self.getNumWordVectors()
		
		size = len(self.wordVectors)
		simArray = np.empty(shape=(size,size))
		for i in range(size):
			simArray[i][i] = 1.0
		
		for i in range(size):
			for j in range(i+1, size):
				if self.similarityAlgo == "cosine":
					sim = cosineSimilarity(self.numWordVectors[i], self.numWordVectors[j])
				elif self.similarityAlgo == "jaccard":
					sim = jaccardSimilarity(self.wordVectors[i], self.wordVectors[j],\
						self.simAlgoNormalizer[0], self.simAlgoNormalizer[1])
				else:
					raise ValueError("invalid similarity algorithms")
				simArray[i][j] = sim
				simArray[j][i] = sim
		return simArray

	def getInterSetSimilarity(self, byCount, normalized, split):
		"""
		inter set pair wise  similarity
		"""
		self.getNumWordVectors()
		size = len(self.wordVectors)
		if not self.similarityAlgo == "jaccard":
			firstNumVec = self.numWordVectors[:split]
			secNumVec = self.numWordVectors[split:]
			fiSize = len(firstNumVec)
			seSize = len(secNumVec)
		else:
			firstVec = self.wordVectors[:split]
			secVec = self.wordVectors[split:]
			fiSize = len(firstVec)
			seSize = len(secVec)
		
		simArray = np.empty(shape=(fiSize,seSize))
		for i in range(fiSize):
			for j in range(seSize):
				if self.similarityAlgo == "cosine":
					sim = cosineSimilarity(firstNumVec[i], secNumVec[j])
				elif self.similarityAlgo == "jaccard":
					sim = jaccardSimilarity(firstVec[i], secVec[j],\
						self.simAlgoNormalizer[0], self.simAlgoNormalizer[1])
				else:
					raise ValueError("invalid similarity algorithms")
				simArray[i][j] = sim
		return simArray

	def getNumWordVectors(self):
		"""
		get vectors
		"""
		if not self.similarityAlgo == "jaccard":
			if self.numWordVectors is None:
				self.numWordVectors = list(map(lambda wv: self.termTable.getVector(wv, byCount, normalized), self.wordVectors))

# clean doc to create term array
def clean(doc, preprocessor, verbose):
	if verbose:
		print ("--raw doc")
		print (doc)
	#print "next clean"
	doc = preprocessor.removeNonAsciiFromText(doc)
	words = preprocessor.tokenize(doc)
	words = preprocessor.allow(words)
	words = preprocessor.toLowercase(words)
	words = preprocessor.removeStopwords(words)
	words = preprocessor.removeShortWords(words, 3)
	words = preprocessor.removePunctuation(words)
	words = preprocessor.lemmatizeWords(words)
	#words = preprocessor.removeNonAscii(words)
	if verbose:
		print ("--after pre processing")
		print (words)
	return words

# get sentences
def getSentences(filePath):
	with open(filePath, 'r') as contentFile:
		content = contentFile.read()
		sentences = content.split('.')
	return sentences


