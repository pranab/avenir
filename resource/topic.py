#!/usr/bin/python

import os
import sys
import nltk
from nltk.corpus import movie_reviews
from random import shuffle
import gensim
from gensim import corpora
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from preprocess import *
from process import *

# number of topics
items = sys.argv[1].split(",")
print "number of topics to be tried " + str(len(items))

if len(items) == 2:
	minTopic = int(items[0])
	maxTopic = int(items[1])
else:
	minTopic =  maxTopic = int(sys.argv[1])

# dcument list
if len(sys.argv) > 2:
	docComplete  = []
	for i in range(2, len(sys.argv)):
		filePath = sys.argv[i]
		with open(filePath, 'r') as contentFile:
			content = contentFile.read()
			docComplete.append(content)
else:
	doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
	doc2 = "My father spends a lot of time driving my sister around to dance practice."
	doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
	doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
	doc5 = "Health experts say that Sugar is not good for your lifestyle."

	# compile documents
	docComplete = [doc1, doc2, doc3, doc4, doc5]

# processor objects
preprocessor = TextPreProcessor()

def clean(doc):
	print "--raw doc"
	print doc
	words = preprocessor.tokenize(doc)
	words = preprocessor.toLowercase(words)
	words = preprocessor.removeStopwords(words)
	words = preprocessor.removePunctuation(words)
	words = preprocessor.lemmatizeWords(words)
	print "--after pre processing"
	print words
	return words

# pre process
docClean = [clean(doc) for doc in docComplete]

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(docClean)
print dictionary

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
docTermMatrix = [dictionary.doc2bow(doc) for doc in docClean]

for numTopics in range(minTopic, maxTopic+1):
	print "**num of topics " + str(numTopics)
	processor = TextProcessor()

	# Running and Trainign LDA model on the document term matrix.
	ldaModel = processor.ldaModel(docTermMatrix, numTopics, dictionary, 100)

	# output
	print "--topics"
	print(ldaModel.print_topics(num_topics=numTopics, num_words=5))

	print "--doc latent vector"
	for doc in docTermMatrix:
		docLda = ldaModel[doc]
		print docLda

	# perplexity
	perplexity = ldaModel.log_perplexity(docTermMatrix)
	print "--perplexity %.6f" %(perplexity)






