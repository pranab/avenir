#!/usr/local/bin/python3

# beymani-python: Machine Learning
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
import random
import statistics 
import numpy as np
import matplotlib.pyplot as plt 
import threading
import time
import queue
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../supv"))
sys.path.append(os.path.abspath("../text"))
from util import *
from sampler import *
from tnn import *
from txproc import *

"""
causality analysis 
"""
emailDoms = ["yahoo.com", "gmail.com", "hotmail.com", "aol.com"]

def mutStr(st):
	"""
	mutate a char in string
	"""
	l = len(st)
	ci = randomInt(0, l - 1)
	cv = st[ci]
	if cv.isdigit():
		r = selectRandomFromList(dig)
	elif cv.isupper():
		r = selectRandomFromList(ucc)
	else:
		r = selectRandomFromList(lcc)
	
	nst = st[:ci] + r + st[ci+1:] if l > 1 else r
	return nst

def createPosMatch(rec, fi):
	"""
	create positive match by mutating a field
	"""
	mrec = rec.copy()
	fv = mrec[fi]
	nc = fv.split()
	le = len(nc)
	if fi == 0:
		#name
		if isEventSampled(50):
			nfv = nc[0] + " " +  selectRandomFromList(ucc) + " " +  nc[1]
		else:
			nc[1] = mutStr(nc[1])
			nfv = nc[0] + " "  + nc[1]
	elif fi == 1:
		#address
		mutated = False
		if isEventSampled(50):
			mutated = True
			s = nc[-1]
			if s == "Rd":
				nc[-1] = "Road"
			elif s == "Ave":
				nc[-1] = "Avenue"
			elif s == "St":
				nc[-1] = "Street"
			elif s == "Dr":
				nc[-1] = "Drive"
			else:
				mutated = False

		if not mutated:
			si = randomInt(0, 1)
			nc[si] = mutStr(nc[si])
		nfv = " ".join(nc)
	
	elif fi == 2:
		#city
		si = randomInt(0, le - 1) if le > 1 else 0
		nc[si] = mutStr(nc[si])
		nfv = " ".join(nc) if le > 1 else nc[0]
	
	elif fi == 3:
		#state
		nc[0] = mutStr(nc[0])
		nfv = nc[0]
		
	elif fi == 4:
		#zip
		nc[0] = mutStr(nc[0])
		nfv = nc[0]

	elif fi == 5:
		#email
		if isEventSampled(60):
			nc[0] = mutStr(nc[0])
			nfv = nc[0]
		else:
			nfv = genLowCaseID(randomInt(4, 10)) + "@" + selectRandomFromList(emailDoms)
	
	mrec[fi] = nfv
	return  mrec

def printNgramVec(ngv):
	"""
	print ngram vector
	"""
	print("ngram vector")
	for i in range(len(ngv)):
		if ngv[i] > 0:
			print("{} {}".format(i, ngv[i]))
			
def createNegMatch(tdata, ri):
	"""
	create negative match by randomly selecting another record
	"""
	nri = randomInt(0, len(tdata)-1)
	while nri == ri:
		nri = randomInt(0, len(tdata)-1)
	return tdata[nri]


def createNgramCreator():
	""" create ngram creator """
	cng = CharNGram(["lcc", "ucc", "dig"], 3, True)
	spc = ["@", "#", "_", "-", "."]
	cng.addSpChar(spc)
	cng.setWsRepl("$")
	cng.finalize()
	return cng
	
		
def getSim(rec, incOutput=True):
	""" get rec pair similarity """
	#print(rec)
	sim = list()
	for i in range(6):
		#print("field " + str(i))
		if i == 3:
			s = levenshteinSimilarity(rec[i],rec[i+6])
		else:
			ngv1 = cng.toMgramCount(rec[i])
			ngv2 = cng.toMgramCount(rec[i+6])
			#printNgramVec(ngv1)
			#printNgramVec(ngv2)
			s = cosineSimilarity(ngv1, ngv2)
		sim.append(s)
	ss = toStrFromList(sim, 6)
	srec = ss + "," + rec[-1] if incOutput else ss
	return srec
	
class SimThread (threading.Thread):
	""" multi threaded similarity calculation """
	
	def __init__(self, tName, cng, qu, incOutput, outQu, outQuSize):
		""" initialize """
		threading.Thread.__init__(self)
		self.tName = tName
		self.cng = cng
		self.qu = qu
		self.incOutput = incOutput
		self.outQu = outQu
		self.outQuSize = outQuSize

	def run(self):
		""" exeution """
		while not exitFlag:
			rec = dequeue(self.qu, workQuLock)
			if rec is not None:
				srec = getSim(rec, self.incOutput)
				if outQu is None:
					print(srec)
				else:
					enqueue(srec, self.outQu, outQuLock, self.outQuSize)
			
def createThreads(nworker, cng, workQu, incOutput, outQu, outQuSize):
	"""create worker threads """
	threadList = list(map(lambda i : "Thread-" + str(i+1), range(nworker)))
	threads = list()
	for tName in threadList:
		thread = SimThread(tName, cng, workQu, incOutput, outQu, outQuSize)
		thread.start()
		threads.append(thread)
	return threads


def enqueue(rec, qu, quLock, qSize): 
	""" enqueue record """
	queued = False
	while not queued:
		quLock.acquire()
		if qu.qsize() < qSize - 1:
			qu.put(rec)
			queued = True
		quLock.release()
		time.sleep(1)

def dequeue(qu, quLock): 
	""" dequeue record """
	rec = None
	quLock.acquire()
	if not qu.empty():
		rec = qu.get()
	quLock.release()
	
	return rec
	
if __name__ == "__main__":
	op = sys.argv[1]

	#multi threading related
	workQuLock = threading.Lock()
	outQuLock = threading.Lock()
	exitFlag = False
	

	if op == "gen":
		""" generate data from from source file"""
		srcFilePath = sys.argv[2]
		i = 0
		for rec in fileRecGen(srcFilePath, ","):
			if i > 0:
				nrec = list()
				fname = rec[0][1:-1]
				lname = rec[1][1:-1]
				nrec.append(fname + " " + lname)
				nrec.append(rec[-9][1:-1])
				nrec.append(rec[-8][1:-1])
				nrec.append(rec[-6][1:-1])
				z = rec[-5]
				if len(z) == 7:
					z = z[1:-1]
				nrec.append(z)
				nrec.append(rec[-2][1:-1])
				print(",".join(nrec))
			i += 1
			
	if op == "genad":
		""" generate additional data by swapping name and address with another random record"""
		srcFilePath = sys.argv[2]
		nrec = int(sys.argv[3])
		tdata = getFileLines(srcFilePath)
		
		for _ in range(nrec):
			r1 = selectRandomFromList(tdata)
			#print(",".join(r1))
			r2 = selectRandomFromList(tdata)
			while r1[0] == r2[0]:
				r1 = selectRandomFromList(tdata)
				r2 = selectRandomFromList(tdata)
			
			nm =  r2[0]
			r1[0] = nm
			r1[1] = r2[1]
			email = nm.split()[0].lower() + "@" + r1[5].split("@")[1]
			r1[5] = email
			print(",".join(r1))

	if op == "gendup":
		""" replace some records in first file  with reccords from another file"""
		srcFilePath = sys.argv[2]
		dupFilePath = sys.argv[3]
		ndup = int(sys.argv[4])
		
		tdata = getFileLines(srcFilePath, None)
		
		percen = 10
		tdataSec =  list()
		while len(tdataSec) < ndup:
			tdataSec = getFileSampleLines(dupFilePath, percen)
			percen = int(percen * ndup / len(tdataSec) + 2)
		
		tdataSec = selectRandomSubListFromList(tdataSec, ndup)
		drecs = list()
		for rec in tdataSec:
			fi = randomInt(0, 5)
			mrec = createPosMatch(rec, fi)
			if isEventSampled(30):
				fi = randomInt(0, 5)
				mrec = createPosMatch(mrec, fi)
			drecs.append(",".join(mrec))
			
		setListRandomFromList(tdata, drecs)
		for r in tdata:
			print(r)

	elif op == "genpn":
		""" generate pos pos and pos neg paire """
		srcFilePath = sys.argv[2]
		tdata = getFileLines(srcFilePath, None) if len(sys.argv) == 3 else getFileLines(sys.argv[3], None)
		
		ri = 0
		for rec in fileRecGen(srcFilePath, ","):
			for _ in range(2):
				fi = randomInt(0, 5)
				mrec = createPosMatch(rec, fi)
				if isEventSampled(30):
					fi = randomInt(0, 5)
					mrec = createPosMatch(mrec, fi)
				print(",".join(rec) + "," + ",".join(mrec) + "," + "1")
				
			for _ in range(2):
				mrec = createNegMatch(tdata, ri)
				print(",".join(rec) + "," + mrec + "," + "0")
				
			ri += 1

	elif op == "sim":
		""" create field pair similarity """
		srcFilePath = sys.argv[2]
		cng = CharNGram(["lcc", "ucc", "dig"], 3, True)
		spc = ["@", "#", "_", "-", "."]
		cng.addSpChar(spc)
		cng.setWsRepl("$")
		cng.finalize()
		c = 0
		for rec in fileRecGen(srcFilePath, ","):
			#print(",".join(rec))
			srec = getSim(rec)
			print(srec)
			c += 1
				
	elif op == "msim":
		""" create field pair similarity in parallel"""
		srcFilePath = sys.argv[2]
		nworker = int(sys.argv[3])
		
		cng = createNgramCreator()
		c = 0
		
		#create threads
		qSize = 100
		workQu = queue.Queue(qSize)
		threads = createThreads(nworker, cng, workQu, True, None, None)
			
		for rec in fileRecGen(srcFilePath, ","):
			enqueue(rec, workQu, qSize)
		
		#wrqp up
		while not workQu.empty():
			pass
		exitFlag = True
		for t in threads:
			t.join()
			
	elif op == "nnTrain":
		""" train neural network model """
		prFile = sys.argv[2]
		regr = FeedForwardNetwork(prFile)
		regr.buildModel()
		FeedForwardNetwork.batchTrain(regr)
				
	elif op == "nnPred":
		""" predict with neural network model """
		newFilePath = sys.argv[2]
		existFilePath = sys.argv[3]
		nworker = int(sys.argv[4])
		prFile = sys.argv[5]
		
		regr = FeedForwardNetwork(prFile)
		regr.buildModel()
		cng = createNgramCreator()

		#create threads
		qSize = 100
		workQu = queue.Queue(qSize)
		outQu = queue.Queue(qSize)
		threads = createThreads(nworker, cng, workQu, False, outQu, qSize)
		
		for nrec in fileRecGen(newFilePath):
			srecs = list()
			ecount = 0
			#print("processing ", nrec)
			for erec in fileRecGen(existFilePath):
				rec = nrec.copy()
				rec.extend(erec)
				#print(rec)
				enqueue(rec, workQu, workQuLock, qSize)
				srec = dequeue(outQu, outQuLock)
				if srec is not None:
					srecs.append(strToFloatArray(srec))
				ecount += 1
			
			#wait til workq queue is drained
			while not workQu.empty():
				pass
			
			#drain out queue	
			while len(srecs) < ecount:
				srec = dequeue(outQu, outQuLock)
				if srec is not None:
					srecs.append(strToFloatArray(srec))
				time.sleep(1)
				
			#predict
			simMax = 0
			sims = FeedForwardNetwork.predict(regr, srecs)
			sims = sims.reshape(sims.shape[0])
			print("{}  {:.3f}".format(nrec, max(sims)))
			
		exitFlag = True
		for t in threads:
			t.join()

	
