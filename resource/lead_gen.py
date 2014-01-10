#!/usr/bin/python

import sys
import redis
import uuid
import thread
import time
import random

r = redis.StrictRedis(host='localhost', port=6379, db=0)

actionSel = {'page1' : 0, 'page2' : 0, 'page3' : 0}
actionCtrDistr = {'page1' : (30, 12), 'page2' : (60, 30), 'page3' : (80, 10)}
actionSelCountThreshold = 50


# send event, for example page visit
def sendEvent(intv):
	print "starting event thread" 
	rc = redis.StrictRedis(host='localhost', port=6379, db=0)
	roundNum = 1
	eventCount = 0
	while True:
		sessionID = uuid.uuid1()
		msg = "%s,%d" %(sessionID, roundNum)
		rc .lpush("eventQueue", msg)
		roundNum = roundNum + 1
		eventCount = eventCount + 1;
		if (eventCount % 1000 == 0):
			print "generated %d events" % (eventCount)
		intv = intv * (float(random.randrange(80, 120)) / 100.0)
		time.sleep(intv)

# receive action, for example page to be displayed
def receiveAction(intv):
	print "starting action thread"
	rc = redis.StrictRedis(host='localhost', port=6379, db=0)
	actionCount = 0		
	while True:
		data = rc.rpop("actionQueue")
		if data is not None:
			#print data
			action = data.split(",")[1]
			updateClickRate(rc,action)
			actionCount = actionCount + 1
			if (actionCount % 1000 == 0):
				print "got %d actions" % (actionCount)
		time.sleep(intv)

#update CTR when enough samples are available 
def updateClickRate(rc,action):
	actionSel[action] = actionSel[action] + 1             
	if (actionSel[action] == actionSelCountThreshold):
		distr = actionCtrDistr[action]
		sum = 0
		for i in range(12):
			sum = sum + random.randrange(1, 100)
		r = float(sum - 600) / 100.0
		r = int(r * distr[1] + distr[0])
		if (r < 0):
			r = 0
		actionSel[action] = 0
		ctr = "%s,%d" % (action, r)
		rc.lpush("rewardQueue", ctr)      
		print "action %s ctr %d" %(action, r)      

def flushQueue(queue):
	count = 0
	while True:
		out = r.rpop(queue)
		if out is not None:
			print out
			count = count + 1
		else:
			break
	print "%d items flushed from %s" %(count, queue)

def queueLength(queue):
	qlen = r.llen(queue)
	print "queue %s has %d items" %(queue, qlen)		

def browseQueue(queue):
	start = 0
	while True:
		items =r.lrange(queue, start, start+1000000) 
		count = 0
		for item in items:
			print item
			count = count + 1
		if (count > 0):
			start = start + count
		else:
			break
			
			
#commands	
op = sys.argv[1]
if (op == "simulate"):	            
	eventIntv = float(sys.argv[2])

	try:
	   thread.start_new_thread(sendEvent, (eventIntv, ))
	   thread.start_new_thread(receiveAction, (eventIntv / 10, ))
	except:
	   print "Error: unable to start thread"

	com = "continue"
	while (com == "continue"):
		com = raw_input("enter quit to quit\n")
		pass
elif (op == "flushQueue"):
	flushQueue("eventQueue")
	flushQueue("actionQueue")
	flushQueue("rewardQueue")
elif (op == "queueLength"):
	queueLength("eventQueue")
	queueLength("actionQueue")
	queueLength("rewardQueue")
elif (op == "browseQueue"):
	browseQueue("eventQueue")
	browseQueue("actionQueue")
	browseQueue("rewardQueue")
