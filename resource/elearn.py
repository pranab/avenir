#!/usr/bin/python

import os
import sys
import random 
import time
import uuid
import threading
import math
sys.path.append(os.path.abspath("../lib"))
from sampler import *

numUsers = int(sys.argv[1])
samplers = {
	'contentTime' : GaussianRejectSampler(300, 100),
	'discussTime' : GaussianRejectSampler(80, 40),
	'organizerTime' : GaussianRejectSampler(40, 20),
	'emailCount' : GaussianRejectSampler(10, 6),
	'testScore' : GaussianRejectSampler(50, 30),
	'assignmentScore' : GaussianRejectSampler(60, 40),
	'chatMsgCount' : GaussianRejectSampler(100, 60),
	'searchTime' : GaussianRejectSampler(60, 40),
	'bookMarkCount' : GaussianRejectSampler(12, 8)
}

for i in range(numUsers):
	fields = []
	failProb = 10
	
	userId = 1000000 + random.randint(0, 1000000)
	
	contentTime = samplers['contentTime'].sample()
	contentTime = minLimit(contentTime, 0)
	fields.append(contentTime)
	if (contentTime < 100):
		failProb += 10
	elif (contentTime < 150):
		failProb += 6
	
	discussTime = samplers['discussTime'].sample()
	discussTime = minLimit(discussTime, 0)
	fields.append(discussTime)
	if (discussTime < 30):
		failProb += 8
	elif (discussTime < 50):
		failProb += 4

	organizerTime = samplers['organizerTime'].sample()
	organizerTime = minLimit(organizerTime, 0)
	fields.append(organizerTime)
	if (discussTime < 10):
		failProb += 5

	emailCount = samplers['emailCount'].sample()
	emailCount = minLimit(emailCount, 0)
	fields.append(emailCount)
	if (emailCount < 3):
		failProb += 6

	testScore = samplers['testScore'].sample()
	testScore = rangeLimit(testScore, 10, 100)
	fields.append(testScore)
	if (testScore < 30):
 		failProb += 34
	elif (testScore < 40):
 		failProb += 20	
	elif (testScore < 50):
 		failProb += 14	

	assignmentScore = samplers['assignmentScore'].sample()
	assignmentScore = rangeLimit(assignmentScore, 10, 100)
	fields.append(assignmentScore)
	if (assignmentScore < 35):
		failProb += 28
	elif (assignmentScore < 50):
		failProb += 18
	elif (assignmentScore < 60):
		failProb += 10

	chatMsgCount = samplers['chatMsgCount'].sample()
	chatMsgCount = minLimit(chatMsgCount, 0)
	fields.append(chatMsgCount)
	if (chatMsgCount < 20):
		failProb += 4

	searchTime = samplers['searchTime'].sample()
	searchTime = minLimit(searchTime, 0)
	fields.append(searchTime)
	if (searchTime < 15):
		failProb += 7
	elif (searchTime < 30):
		failProb += 3

	bookMarkCount = samplers['bookMarkCount'].sample()
	bookMarkCount = minLimit(bookMarkCount, 0)
	fields.append(bookMarkCount)
	if (bookMarkCount < 4):
		failProb += 8

	status = "P"
	if random.randint(0, 100) < failProb:
		status = "F"
	msg = ",".join(map(str, fields))
	msg = "%d,%s,%s" %(userId,msg,status)
	print msg
	