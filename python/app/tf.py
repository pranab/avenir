#!/usr/bin/python

import os
import sys
import nltk
from nltk.corpus import movie_reviews
from random import shuffle
import matplotlib.pyplot as plt
import jprops
from sklearn.datasets import fetch_20newsgroups
import pickle
import math
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from preprocess import *
from util import *
from mlutil import *

# base term ditsribution
def crateTermDistr(docs, vocFilt, saveFile):
	print "term distribution"
	print "num of docs " + str(len(docs))
	tfidf = TfIdf(vocFilt, False)
	for d in docs:
		tfidf.countDocWords(d)
	tf = tfidf.getWordFreq()
	print "term distr size " + str(len(tf))
	verbose = False
	if verbose:
		for t in tf.items():
			print "%s	%.6f" %(t[0], t[1])
	
	if saveFile is not None:
		tfidf.save(saveFile)
	return tfidf

# load term distr from file
def loadTermDistr(saveFile):
	tfidf = TfIdf.load(saveFile)
	return tfidf

###############################################################################
preprocessor = TextPreProcessor()
mode = sys.argv[1]
verbose = False

if mode == "buildBaseTf":
	saveFile = sys.argv[2]
	categories = ['rec.autos', 'soc.religion.christian','comp.graphics', 'sci.med', 'talk.politics.misc', 'misc.forsale', 'sci.electronics', 'rec.sport.baseball', 'talk.religion.misc']

	twentyTrain = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
	print "pre processing " + str(len(twentyTrain.data)) + " docs"
	docsClean = [clean(doc, preprocessor, verbose) for doc in twentyTrain.data]	
	crateTermDistr(docsClean, None, saveFile)

elif mode == "loadBaseTf":
	saveFile = sys.argv[2]
	tfidf = loadTermDistr(saveFile)

elif mode == "tfDiff":
	dataDir = sys.argv[2]
	basTfFile = sys.argv[3]
	baseTf = loadTermDistr(basTfFile)
	baseDistr = baseTf.getWordFreq()

	docs, filePaths  = getFileContent(dataDir, True)
	docsClean = [clean(doc, preprocessor, verbose) for doc in docs]	
	thisTf = crateTermDistr(docsClean, None, None)
	thisDistr = thisTf.getWordFreq()
	
	# aggregate vocabulary
	vocab = baseTf.getVocabulary()
	vocab.update(thisTf.getVocabulary())
	
	filt = []
	th = 0.0000001
	rent = 0.0
	for w in vocab:
		p = thisDistr.get(w, 0)
		q = baseDistr.get(w, 0)
		if  p > 0:
			if q > 0:
				rent  = p * math.log(p / q)
			else:
				rent = 1000.0
		else:
			if q > th:
				rent  = -1000.0
			else:
				rent = 0.0

		if rent > 0:
			filt.append((w, rent, p))

	filt.sort(key=takeThird, reverse=True)
	print "filtered vocab size " + str(len(filt))
	print "showing top 100"
	for f in filt[:100]:
		print f


