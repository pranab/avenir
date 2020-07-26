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

# Package imports
import os
import sys
import spacy
import torch
import numpy
import statistics 
import math
from numpy.testing import assert_almost_equal

"""
Semantic search with BERT pre trained transformer model. Embedding at various levels are used,
document, sentence and words
"""
def wordVectorList(doc):
	tstr = doc._.trf_word_pieces_
	for st, ve in zip(tstr, doc._.trf_last_hidden_state):
		print(st + " " + str(ve[:16]))
		
def docAv(qu, doc):
	"""
	whole doc similarity
	"""
	return qu.similarity(doc)

def tokSimMax(qu, doc):
	"""
	token pair wise max
	"""
	msi = 0
	for t1 in qu:
		for t2 in doc:
			si = t1.similarity(t2)
			if si > msi:
				msi = si
	return msi
	
def tokSimAvMax(qu, doc):
	"""
	token max then average
	"""
	qts = list()
	for t1 in qu:
		msi = 0
		for t2 in doc:
			si = t1.similarity(t2)
			if not math.isnan(si) and si > msi:
				msi = si
		qts.append(msi)
		
	amsi = numpy.mean(numpy.array(qts))
	return amsi
		
def tokSimMaxAv(qu, doc):
	"""
	token average and then max
	"""
	masi = 0
	for t1 in qu:
		qts = list()
		for t2 in doc:
			si = t1.similarity(t2)
			if not math.isnan(si):
				qts.append(si)
		av = numpy.mean(numpy.array(qts))
		#print(len(qts))
		#print(av)
		if av > masi:
			masi = av
	return masi

def tokSimAv(qu, doc):
	"""
	token pair wise average
	"""
	qts = tokSim(qu, doc)
	asi = numpy.mean(qts)	
	return asi

def tokSimMed(qu, doc):
	"""
	token pair wise average
	"""
	qts = tokSim(qu, doc)
	asi = numpy.median(qts)	
	return asi

def tokSim(qu, doc):
	"""
	token pair similarity
	"""
	qts = list()
	for t1 in qu:
		for t2 in doc:
			si = t1.similarity(t2)
			if not math.isnan(si):
				qts.append(si)
	return numpy.array(qts)
	
def sentSimAv(qu, doc):
	"""
	sentence wise average
	"""
	sil = sentSim(qu, doc)
	asi = numpy.mean(sil)			
	return asi	

def sentSimMed(qu, doc):
	"""
	sentence wise median
	"""
	sil = sentSim(qu, doc)
	asi = numpy.median(sil)			
	return asi	

def sentSimMax(qu, doc):
	"""
	sentence wise average
	"""
	sil = sentSim(qu, doc)
	asi = numpy.max(sil)			
	return asi	

def sentSim(qu, doc):
	"""
	sentence similarity
	"""
	qe = qu._.trf_last_hidden_state[0]
	dstr = doc._.trf_word_pieces_
	dte = zip(dstr, doc._.trf_last_hidden_state)
	sil = list()
	for t, v in dte:
		if t == "[CLS]":
			si = 20 - numpy.linalg.norm(qe - v)
			sil.append(si)
	return numpy.array(sil)

def printAlgo(algo):
	"""
	print similarity algo
	"""
	if algo == "ds":
		print("using doc avrage similarity")
	elif algo == "tsma":
		print("using token max similarity")
	elif algo == "tsavm":
		print("using token average of max similarity")
	elif algo == "tsmav":
		print("using token max of average similarity")
	elif algo == "tsa":
		print("using token average similarity")
	elif algo == "tsme":
		print("using token median similarity")
	elif algo == "ssa":
		print("using sentence average similarity")
	elif algo == "ssme":
		print("using sentence median similarity")
	elif algo == "ssma":
		print("using sentence max similarity")
	else:
		raise ValueError("invalid semilarity algo")
		
if __name__ == "__main__":
	algo = sys.argv[1]
	printAlgo(algo)

	is_using_gpu = spacy.prefer_gpu()
	if is_using_gpu:
		torch.set_default_tensor_type("torch.cuda.FloatTensor")

	doc1 = """Apple turned in a record holiday quarter in 2019 thanks to strong demand for the 
	iPhone 11, and it looks like that trend has continued in 2020. According to a report from Omdia, 
	the iPhone 11 managed to steal the title of "world's most popular smartphone" from the iPhone XR - 
	which according to CounterPoint Research was selling like hotcakes to people who are looking 
	to get the most iPhone for their dollar. Omdia says that Apple shipped 19.5 million iPhone 11s 
	in the first quarter of this year, thanks to its relative affordability compared to the iPhone 11 
	Pro and iPhone 11 Pro Max, as well as being a well-rounded iPhone for most people. 
	This is even more impressive when you consider that it achieved these numbers despite global 
	smartphone shipments experiencing their biggest decline in history in March.
	"""
	
	doc2 = """The first brand name that comes to mind when we think of smartphones is, of course 
	the iPhone. Apple, the maker of iPhones, can rightly be said to be one of the pioneers of 
	the smartphone. Apple earned a strong brand recognition for the iPhone and had strong 
	hegemony over the smartphones market for several years after the first iPhone was introduced 
	in 2007. The iPhone has always recorded more than 15% market share for most of its tenure with 
	occasional peaks crossing 20%. The company that saw a very steep growth in the early years was 
	Samsung whose market share rose from 3.3% in Q4 2009 to 29.1% in Q4 2012. The market share has 
	remained more or less stagnant around 20% after that with occasional highs and lows. During the 
	same period Nokia witnessed an equally steep decline from a market share of 38.1% in Q4 2009 to 
	almost 3% in Q4 2012. Some new players that emerged and are presently on a growth trajectory are 
	Huawei, Xiaomi and OPPO.    
	"""
	
	doc3 = """The smartphone industry, as a whole, has been witnessing a decline in shipments for the 
	past three years and the decline continued in 2019. The sales volume of smartphones in H1 2019 was 
	lower by 4% than that in H1 2018. In 2019, there have been some interesting twists in the global 
	smartphone industry. The leader in smartphone innovation, Apple witnessed its market share, in terms of 
	sales volume, drop down to 11.9% in Q1 2019 and 10.1% in Q2 2019. While the market share of Samsung 
	remained more or less consistent, the three emerging players saw their market share going up significantly 
	in 2019. Huawei, in particular, was the winner amongst the emerging leaders with a market share of 19% 
	in Q1 2019 and 15.8% in Q2 2019 in terms of sales volume and with this the market share of Huawei has 
	surpassed Apple in 2019. Xiaomi and OPPO saw their market share reaching 10% and 9% respectively in 
	2019. In fact, the YoY volume drop stood at 30.3% for Apple in Q1 2019 and 13.8% in Q2 2019. On the 
	contrary, the corresponding YoY increase was 50.4% for Huawei in Q1 2019 and 16.5% in Q2 2019. The 
	growth in sales volume of Huawei in Q2 2019 was despite the ban imposed by the United States against 
	its sale in the country.   
	"""
	
	doc4 = """According to IDC, the smartphone sales are expected to be lower in the rest of 2019, 
	compared to 2018. 2019 will be the third consecutive year of fall in smartphone shipments. China 
	and Brazil are the only countries, among the top five countries for smartphone demand, that are 
	expected to close the year positively. The smartphone market has reached saturation in the 
	developed countries and the growth rates are also relatively declining in the developing countries. 
	This could be the reason for fall in sales volume for the last three years. In addition, the 
	years 2018 and 2019 have witnessed growing trade wars. Large number of developed and developing 
	economies have witnessed a recessionary environment in last few years while some are continuing 
	to record subdued economic growth. This has also affected consumer debt amidst low growth in 
	income alongside weakening employment situation. These economic factors could have possibly 
	decreased the willingness among people to pay a high price for smartphones. Also, people 
	change phones when the market has something advanced to offer. The 4G technology has been 
	around for a while and the 5G technology is still in a very incipient stage and is expected 
	to reach the market in 2020. People with existing 4G handsets are perhaps unwilling to buy 
	another 4G handset in 2019 when they know that the 5G handset would be available soon. After 
	the companies release new models with 5G enabled features we may expect a surge in smartphone 
	sales again.   
	"""
	
	doc5 = """In 2016, the number of smartphones sold to consumers stood at around 1.5 billion 
	units, a significant increase from the 680 million units sold in 2012. This means that over 
	28 percent of the world’s total population owned a smart device in 2016, a figure that is 
	expected to increase to 37 percent by 2020. In the same year smartphone penetration is set to 
	reach 60.5 percent in North America as well as in Western Europe. This is a large rise in the 
	29.3 percent of people in North America and 22.7 percent of Western Europeans who had smartphones 
	in 2011. n the United States alone, sales of smartphones are projected to be worth around 55.6 billion 
	U.S. dollars in 2017, an increase from 18 billion dollars in 2010. By 2017, it is forecast that almost 
	84 percent of all mobile users in the United States will own a smartphone, an increase from the 27 
	percent of mobile users in 2010.  
	"""
	
	doc6 = """Samsung released its first Android smartphone in 2009, and can be credited with the launch of 
	the first Android tablet back in 2010. The company is among the biggest players in the smartphone market 
	in the world. It has recently developed smartphones running Tizen OS, as an alternative to its Android-based 
	smartphones. Samsung's latest mobile launch is the Galaxy Z Flip 5G. The Samsung Galaxy Z Flip 5G is powered by 
	2.95GHz octa-core Qualcomm Snapdragon 865+ processor and it comes with 8GB of RAM. The phone packs 256GB of 
	internal storage cannot be expanded.	
	"""
	
	doc7 = """Apples are among the world’s most popular fruits. They grow on the apple tree (Malus domestica), 
	originally from Central Asia. Apples are high in fiber, vitamin C, and various antioxidants. They are also 
	very filling, considering their low calorie count. Studies show that eating apples can have multiple 
	benefits for your health . Usually eaten raw, apples can also be used in various recipes, juices, 
	and drinks. Various types abound, with a variety of colors and sizes. Apples are mainly composed of 
	carbs and water. They’re rich in simple sugars, such as fructose, sucrose, and glucose.Despite 
	their high carb and sugar contents, their glycemic index (GI) is low, ranging 29–44.
	"""

	doc8 = """Apples are loaded with vitamin C, especially in the skins, which are also full of fiber, 
	Flores said. Apples contain insoluble fiber, which is the type of fiber that doesn't absorb water. 
	It provides bulk in the intestinal tract and helps food move quickly through the digestive system, 
	according to Medline Plus.  
	"""
	
	doc9 = """In addition to digestion-aiding insoluble fiber, apples have soluble fiber, such as pectin. 
	This nutrient helps prevent cholesterol from building up in the lining of blood vessels, which, in turn, 
	helps prevent atherosclerosis and heart disease. In a 2011 study, women who ate about 75 grams (2.6 ounces, 
	or about one-third of a cup) of dried apples every day for six months had a 23 percent decrease in bad 
	LDL cholesterol.	  
	"""
	
	doc10 = """One small peach has 12 grams of carbohydrates, 2 grams of fiber, and 11 grams of sugar. Peaches 
	are a low-glycemic fruit, which means they have a minimal effect on blood sugar. Peaches' glycemic index is 
	28 and their glycemic load is 4, putting them in the low range for both GI and GL Peaches contain several 
	important micronutrients, including vitamin C, vitamin A, vitamin K, and B-complex vitamins like thiamin, 
	niacin, and riboflavin. The fruit also provides 247 milligrams of potassium, which is 7% of your 
	recommended daily needs.
	"""
	
	dnames = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
	dtexts = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10]
	
	print("loading BERT transformer model")
	nlp = spacy.load("en_trf_bertbaseuncased_lg")
	print("encoding all docs")
	dvecs = list(map(lambda d : nlp(d), dtexts))
	kvecs = dict(zip(dnames, dvecs))
	
	while True:
		inp = input("enter a search query or q to quit: ")
		if inp == "q":
			break
		vinp = nlp(inp)
		print("\nsearch result")
		res = list()
		for dn, dv in kvecs.items():
			if algo == "ds":
				si = docAv(vinp, dv)
			elif algo == "tsma":
				si = tokSimMax(vinp, dv)
			elif algo == "tsavm":
				si = tokSimAvMax(vinp, dv)
			elif algo == "tsmav":
				si = tokSimMaxAv(vinp, dv)
			elif algo == "tsa":
				si = tokSimAv(vinp, dv)
			elif algo == "tsme":
				si = tokSimMed(vinp, dv)
			elif algo == "ssa":
				si = sentSimAv(vinp, dv)
			elif algo == "ssme":
				si = sentSimMed(vinp, dv)
			elif algo == "ssma":
				si = sentSimMax(vinp, dv)
			else:
				raise ValueError("invalid semilarity algo")
			
			print("{} score {:.6f}".format(dn, si))
			r = (dn, si)
			res.append(r)
		
		res.sort(key=lambda r : r[1], reverse=True)
		print("\nsorted search result")
		for r in res:
			print("{} score {:.6f}".format(r[0], r[1]))
		
		
		
		
	
