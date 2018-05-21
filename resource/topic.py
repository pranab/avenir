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

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
docComplete = [doc1, doc2, doc3, doc4, doc5]

preprocessor = TextPreProcessor()
def clean(doc):
	print doc
	words = preprocessor.tokenize(doc)
	words = preprocessor.toLowercase(words)
	words = preprocessor.removeStopwords(words)
	words = preprocessor.removePunctuation(words)
	words = preprocessor.lemmatizeWords(words)
	print "after pre processing"
	print words
	return words

# pre process
docClean = [clean(doc) for doc in docComplete]


# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(docClean)
print dictionary

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
docTermMatrix = [dictionary.doc2bow(doc) for doc in docClean]

# Running and Trainign LDA model on the document term matrix.
ldamodel = gensim.models.ldamodel.LdaModel(docTermMatrix, num_topics=4, id2word = dictionary, passes=50)

# output
print "topic modeling output"
print(ldamodel.print_topics(num_topics=4, num_words=3))

docLda = ldamodel[docTermMatrix[0]]
print "latent vector"
print docLda








