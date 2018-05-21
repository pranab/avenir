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

#text preprocessor
class TextProcessor:
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

