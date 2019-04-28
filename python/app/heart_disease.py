#!/usr/bin/python

import os
import sys
from random import randint
import time
sys.path.append(os.path.abspath("../lib"))
from util import *
from sampler import *

numSample = int(sys.argv[1])
diseaseDistr = CategoricalRejectSampler(("Y", 25), ("N", 75))

featCondDister = {}

#sex
key = ("Y", 0)
distr = CategoricalRejectSampler(("M", 60), ("F", 40))
featCondDister[key] = distr
key = ("N", 0)
distr = CategoricalRejectSampler(("M", 50), ("F", 50))
featCondDister[key] = distr

#age
key = ("Y", 1)
distr = NonParamRejectSampler(30, 10, 10, 20, 35, 60, 90)
featCondDister[key] = distr
key = ("N", 1)
distr = NonParamRejectSampler(30, 10, 15, 20, 25, 30, 30)
featCondDister[key] = distr

#weight
key = ("Y", 2)
distr = GaussianRejectSampler(190, 8)
featCondDister[key] = distr
key = ("N", 2)
distr = GaussianRejectSampler(150, 15)
featCondDister[key] = distr

#systolic blood pressure
key = ("Y", 3)
distr = NonParamRejectSampler(100, 10, 20, 25, 25, 30, 35, 45, 60, 75)
featCondDister[key] = distr
key = ("N", 3)
distr = NonParamRejectSampler(100, 10, 20, 30, 40, 20, 12, 8, 6, 4)
featCondDister[key] = distr

#dialstolic  blood pressure
key = ("Y", 4)
distr = NonParamRejectSampler(60, 10, 20, 20, 25, 35, 50, 70)
featCondDister[key] = distr
key = ("N", 4)
distr = NonParamRejectSampler(60, 10, 20, 20, 25, 18, 12, 7)
featCondDister[key] = distr

#smoker
key = ("Y", 5)
distr = CategoricalRejectSampler(("NS", 20), ("SS", 35), ("SM", 60))
featCondDister[key] = distr
key = ("N", 5)
distr = CategoricalRejectSampler(("NS", 40), ("SS", 20), ("SM", 15))
featCondDister[key] = distr

#diet
key = ("Y", 6)
distr = CategoricalRejectSampler(("BA", 60), ("AV", 35), ("GO", 20))
featCondDister[key] = distr
key = ("N", 6)
distr = CategoricalRejectSampler(("BA", 15), ("AV", 40), ("GO", 45))
featCondDister[key] = distr

#physical activity per week
key = ("Y", 7)
distr = GaussianRejectSampler(5, 1)
featCondDister[key] = distr
key = ("N", 7)
distr = GaussianRejectSampler(15, 2)
featCondDister[key] = distr

#education
key = ("Y", 8)
distr = GaussianRejectSampler(11, 2)
featCondDister[key] = distr
key = ("N", 8)
distr = GaussianRejectSampler(17, 1)
featCondDister[key] = distr

#ethnicity
key = ("Y", 9)
distr = CategoricalRejectSampler(("WH", 30), ("BL", 40), ("SA", 50), ("EA", 20))
featCondDister[key] = distr
key = ("N", 9)
distr = CategoricalRejectSampler(("WH", 50), ("BL", 20), ("SA", 16), ("EA", 20))
featCondDister[key] = distr

sampler = AncestralSampler(diseaseDistr, featCondDister, 10)

for i in range(numSample):
	(claz, features) = sampler.sample()
	features[2] = int(features[2])
	features[7] = int(features[7])
	features[8] = int(features[8])
	strFeatures = [toStr(f, 3) for f in features]
	print ",".join(strFeatures) + "," + claz
	
	
