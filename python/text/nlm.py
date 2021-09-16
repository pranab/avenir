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

import os
import sys
from random import randint
import random
import time
from datetime import datetime
import re, string, unicodedata
import spacy
import torch
from collections import defaultdict
import pickle
import numpy 
import re
from sentence_transformers import CrossEncoder
import statistics

sys.path.append(os.path.abspath("../lib"))
from util import *
from mlutil import *

"""
neural language model
"""

class NeuralLangModel(object):
	def __init__(self, fragmentor):
		"""
		initialize
		"""
		self.dexts = None
		self.fragmentor = fragmentor
		self.fragments = None
		
	def loadDocs(self, fpaths):
		"""
		loads documents from one file
		"""
		self.fragments = self.fragmentor.generateFragments(fpaths)

#Encoded doc
class EncodedDoc:
	def __init__(self, dtext, dname, drank=None):
		"""
		initialize
		"""
		self.dtext = dtext
		self.dname = dname
		self.drank = drank
		self.denc = None
		self.score = None
		
	def encode(self, nlp):
		"""
		encode
		"""
		self.denc = nlp(self.dtext)

#similarity at token and sentence level for BERT encoding
class SemanticSimilaityBiEnc(NeuralLangModel):
	def __init__(self, fragmentor, docs=None):
		"""
		initialize
		"""
		print("loading BERT transformer model")
		#self.nlp = spacy.load("en_trf_bertbaseuncased_lg")
		self.nlp = spacy.load("en_core_web_lg")
		self.docs = docs if docs is not None else list()
		super(SemanticSimilaityBiEnc, self).__init__(fragmentor)
		
	def docAv(self,qu, doc):
		"""
		whole doc similarity
		"""
		return qu.similarity(doc)
		
	def tokSimAv(self, qu, doc):
		"""
		token pair wise average
		"""
		qts = simAll(qu, doc)
		asi = numpy.mean(qts)	
		return asi
		
	def tokSimMed(self, qu, doc):
		"""
		token pair wise average
		
		"""
		qts = simAll(qu, doc)
		asi = numpy.median(qts)	
		return asi
		
	def tokSimMax(self, qu, doc):
		"""
		token pair wise max (tsma)
		"""
		qte = self. __getTensor(qu)
		dte = self. __getTensor(doc)
		return self.simMax(qte, dte)
		
	def tokSimAvMax(self, qu, doc):
		"""
		token max then average (tsavm)
		"""
		qte = self. __getTensor(qu)
		dte = self. __getTensor(doc)
		return self.simAvMax(qte, dte)

	def tokSimMaxAv(self, qu, doc):
		"""
		token average and then max
		"""
		qte = self. __getTensor(qu)
		dte = self. __getTensor(doc)
		return self.simMaxAv(qte, dte)
		
	def sentSimAv(self, qu, doc):
		"""
		sentence wise average
		"""
		qse, dse = self.__sentEnc(qu, doc)
		sims = self.simAll(qse, dse)
		return numpy.mean(sims)	
		
	def sentSimMed(self, qu, doc):
		"""
		sentence wise average (ssma)
		"""
		qse, dse = self.__sentEnc(qu, doc)
		sims = self.simAll(qse, dse)
		return numpy.median(sims)
		
	def sentSimMax(self, qu, doc):
		"""
		sentence wise average (ssma)
		"""
		qse, dse = self.__sentEnc(qu, doc)
		sims = self.simAll(qse, dse)
		return numpy.maximum(sims)
	

	def sentSimAvMax(self, qu, doc):
		"""
		sentence max then average (tsavm)
		"""
		qse, dse = self.__sentEnc(qu, doc)
		return self.simAvMax(qse, dse)

	def sentSimMaxAv(self, qu, doc):
		"""
		sentence average and then max
		"""
		qse, dse = self.__sentEnc(qu, doc)
		return self.simMaxAv(qse, dse)

	def simMax(self, qte, dte):
		"""
		max similarity between 2 elements
		"""
		msi = 0
		for qt in qte:
			for dt in dte:
				si = cosineSimilarity(qt, dt)
				if not math.isnan(si) and si > msi:
					msi = si
		return msi
	
	def simAvMax(self, qte, dte):
		"""
		max then average (tsavm)
		"""
		qts = list()
		for qt in qte:
			msi = 0
			for dt in dte:
				si = cosineSimilarity(qt, dt)
				if not math.isnan(si) and si > msi:
					msi = si
			qts.append(msi)
		
		amsi = numpy.mean(numpy.array(qts))
		return amsi

	def simMaxAv(self, lqe, lde):
		"""
		average and then max
		"""
		masi = 0
		for qe in lqe:
			qes = list()
			for de in lde:
				si = cosineSimilarity(qe, de)
				if not math.isnan(si):
					qes.append(si)
			av = numpy.mean(numpy.array(qes))
			if av > masi:
				masi = av
		return masi
	
	def simAll(self, lqe, lde):
		"""
		all similarity
		"""
		qes = list()
		for qe in lqe:
			for de in lde:
				si = cosineSimilarity(qe, de)
				if not math.isnan(si):
					qes.append(si)
		return numpy.array(qes)

	def __sentEnc(self, qu, doc):
		"""
		sentence encoding for query and doc
		"""
		qstr = qu._.trf_word_pieces_
		qte = zip(qstr, qu._.trf_last_hidden_state)
		qse = list()	
		for t, v in qte:
			if t == "[CLS]":
				qse.append(v)
	
	
		dstr = doc._.trf_word_pieces_
		dte = zip(dstr, doc._.trf_last_hidden_state)
		dse = list()
		for t, v in dte:
			if t == "[CLS]":
				dse.append(v)
			
		enp = (numpy.array(qse), numpy.array(dse))		
		return enp

	def __getTensor(self, toks):
		"""
		tensors from tokens
		"""
		return list(map(lambda t: t.tensor, toks))
		
	def addEncodedDocs(self, docs):
		"""
		add named doc content
		"""
		self.docs.extend(docs)
	
	def loadFileDocs(self, fpaths):
		"""
		loads documents from one file
		"""
		self.loadDocs(fpaths)
				
		docs = list(map(lambda dnt : EncodedDoc(dnt[1], dnt[0]), self.fragments))
		self.docs.extend(docs)
		
	def search(self, qstr, algo, gdranks=None):
		"""
		tensors from tokens
		"""
		qv = self.nlp(qstr)
		res = list()
		for d in self.docs:
			dn = d.dname
			if d.denc == None:
				d.encode(self.nlp)
			dv = d.denc
			if algo == "ds":
				si = self.docAv(qv, dv)
			elif algo == "tsa":
				si = self.tokSimAv(qv, dv)
			elif algo == "tsme":
				si = self.tokSimMed(qv, dv)
			elif algo == "tsma":
				si = self.tokSimMax(qv, dv)
			elif algo == "tsavm":
				si = self.tokSimAvMax(qv, dv)
			elif algo == "tsmav":
				si = self.tokSimMaxAv(qv, dv)
			elif algo == "ssa":
				si = self.sentSimAv(qv, dv)
			elif algo == "ssme":
				si = self.sentSimMed(qv, dv)
			elif algo == "ssma":
				si = self.sentSimMax(qv, dv)
			elif algo == "ssavm":
				si = self.sentSimAvMax(qv, dv)
			elif algo == "ssmav":
				si = self.sentSimMaxAv(qv, dv)
			else:
				si = -1.0
				print("invalid semilarity algo")
			
			#print("{} score {:.6f}".format(dn, si))
			d.score = si
			r = (dn, si)
			res.append(r)

		#search score for each document
		res.sort(key=lambda r : r[1], reverse=True)
		print("\nsorted search result")
		print("query: {}     matching algo: {}".format(qstr, algo))
		for r in res:
			print("{} score {:.3f}".format(r[0], r[1]))
		
		#rank order if gold truuth rank provided	
		if gdranks is not None:
			i = 0
			count = 0
			for d in gdranks:
				while i < len(gdranks):
					if d == res[i][0]:
						count += 1
						i += 1
						break;
					i += 1
			ro = count / len(gdranks)
			print("rank order {:.3f}".format(ro))	

#similarity at passage or paragraph level using sbertcross encoder
class SemanticSimilaityCrossEnc(NeuralLangModel):

	def __init__(self, fragmentor, docs=None):
		self.dparas = None
		self.scores = None
		print("loading cross encoder")
		self.model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2")
		print("done loading cross encoder")
		super(SemanticSimilaityCrossEnc, self).__init__(fragmentor)
		
	def loadFileDocs(self, fpaths):
		"""
		loads documents from one file
		"""
		self.loadDocs(fpaths)

	def search(self, qstr, algo):
		"""
		returns paragarph pair similarity across 2 documents
		"""
		qdpairs = list(map(lambda dnt : [qstr, dnt[1]], self.fragments))
		qdnt = list(map(lambda dnt : [qstr, dnt[0], dnt[1]],  self.fragments))

		print("fragments")
		for n, t in self.fragments:
			print(n + "\t" +  t[:20])

		print("input shape " + str(numpy.array(qdpairs).shape))
		scores = self.model.predict(qdpairs)
		print("score shape " + str(numpy.array(scores).shape))
		#assertEqual(len(scores), len(self.dparas[0]) * len(self.dparas[1]), "no of scores don't match no of paragraph pairs")
		print(scores)
		
		print("query text fragment pair wise similarity")
		for fs in zip(qdnt, scores):
			print("query: {}\t  doc name: {}\t  doc fragment: {}\t score: {:.6f}".format(fs[0][0][:20], fs[0][1], fs[0][2][:20], fs[1]))
		self.scores = scores
		
		if not self.fragmentor.isDocLevel():
			if algo == "sa":
				agScores = self.aggregateScore("av")
			elif algo == "sme":	
				agScores = self.aggregateScore("med")
			elif algo == "smax":	
				agScores = self.aggregateScore("max")
			else:
				exitWithMsg("invalid matching score aggregator")
		
			print("aggregate doc scores")
			for dn, ags in agScores:
				print("{}\t{:.6f}".format(dn, ags))
			
	def aggregateScore(self, aggr):
		"""
		aggregagte fragment scores for each document
		"""
		dnp = None
		agScores = list()
		scores = list()
		for fs in zip(self.fragments, self.scores):
			dn = fs[0][0].split(":")[0]
			if dnp is not None and dn != dnp:
				ags = self.aggrgeate(aggr, scores)
				ds = (dnp, ags)
				agScores.append(ds)
				scores.clear()
				dnp = dn
			else:
				if dnp is None:
					dnp = dn
				scores.append(fs[1])	
			
		ags = self.aggrgeate(aggr, scores)
		ds = (dnp, ags)
		agScores.append(ds)
		return agScores		
		
	def aggrgeate(self, aggr, scores):
		if aggr == "av":
			ags = statistics.mean(scores)
		elif aggr == "med":
			ags = statistics.median(scores)
		elif aggr == "max":
			ags = max(scores)
		return ags
		
		
def ner(text, nlp):
	#nlp = spacy.load("en_core_web_md")
	doc = nlp(text)
	for ent in doc.ents:
		print(ent.text, ent.start_char, ent.end_char, ent.label_)
		
