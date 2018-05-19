#!/usr/bin/python

import os
import sys
import nltk
from nltk.corpus import movie_reviews
from random import shuffle
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../text"))
from preprocess import *

 
def document_features(document, word_features):
	document_words = set(document)
	features = {}
	for word in word_features:
		features[word] = (word in document_words)
	return features

preprocessor = TextPreProcessor()

documents = [(list(movie_reviews.words(fileid)), category) 
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]
 
shuffle(documents)
#print documents[0]

#all words
all_words = [w.lower() for w in movie_reviews.words()]
print "vocabulary size " + str(len(all_words))
print all_words[:50]

#all_words = preprocessor.removeStopwords(all_words)
#print "vocabulary size after stop word removal " + str(len(all_words))
#print all_words[:50]
 
#all words with counts in corpus
all_words_counts = nltk.FreqDist(w.lower() for w in movie_reviews.words())

#top 2000 words by count
word_features = all_words_counts.keys()[:3000]
#print word_features

#remove stopwords
word_features = preprocessor.removeStopwords(word_features)
print "size after stop word removal " + str(len(word_features))
print word_features[:50]

#stem
word_features = preprocessor.stemWords(word_features)
print "size after stemming " + str(len(word_features))
print word_features[:50]


word_features = word_features[:2000]

# Generate the feature sets for the movie review documents 
featuresets = [(preprocessor.documentFeatures(d, word_features), c) for (d, c) in documents]
#print featuresets[0]

train_set, test_set = featuresets[100:], featuresets[:100]

test_text = "I love this movie, very interesting"
new_test_set = preprocessor.documentFeatures(test_text.split(), word_features)

cls = sys.argv[1]
if cls == "nb":
	#NB classifier
	nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
	acc = nltk.classify.accuracy(nb_classifier, test_set)
	print acc

	#classifier.show_most_informative_features(20)

	print new_test_set
	pred = nb_classifier.classify(new_test_set)
	print pred
elif cls == "me":
	me_classifier = nltk.MaxentClassifier.train(train_set, "megam")
	acc = nltk.classify.accuracy(me_classifier, test_set)
	print acc
	pred = me_classifier.classify(new_test_set)
	print pred
else:
	print "invalid classifier"







