#!/usr/bin/python

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
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tag import StanfordNERTagger

#text preprocessor
class TextPreProcessor:
	def __init__(self, verbose=False):
		pass

def stripHtml(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def removeBetweenSquareBrackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoiseText(text):
    text = stripHtml(text)
    text = removeBetweenSquareBrackets(text)
    return text

def replaceContractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def tokenize(text):
	words = nltk.word_tokenize(text)
	return words

def removeNonAscii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    newWords = []
    for word in words:
        newWord = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        newWords.append(newWord)
    return newWords

def toLowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    newWords = []
    for word in words:
        newWord = word.lower()
        newWords.append(newWord)
    return newWords

def removePunctuation(words):
    """Remove punctuation from list of tokenized words"""
    newWords = []
    for word in words:
        newWord = re.sub(r'[^\w\s]', '', word)
        if newWord != '':
            newWords.append(newWord)
    return newWords

def replaceNumbers(words):
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

def removeStopwords(words):
    """Remove stop words from list of tokenized words"""
    newWords = []
    for word in words:
        if word not in stopwords.words('english'):
            newWords.append(word)
    return newWords

def removeCustomStopwords(words, stopWords):
    """Remove stop words from list of tokenized words"""
    newWords = []
    for word in words:
        if word not in stopWords:
            newWords.append(word)
    return newWords

def stemWords(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatizeVerbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = removeNonAscii(words)
    words = toLowercase(words)
    words = removePunctuation(words)
    words = replaceNumbers(words)
    words = removeStopwords(words)
    return words

def posTag(textTokens):
	tags = nltk.pos_tag(text)
	return tags

def extractEntity(textTokens, classifierPath, jarPath):
	st = StanfordNERTagger(classifierPath, jarPath) 
	entities = st.tag(textTokens)
	return entities


