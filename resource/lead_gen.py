#!/usr/bin/python

import sys
import redis
import uuid
import thread
import time
import random

r = redis.StrictRedis(host='localhost', port=6379, db=0)

actionSel = {'page1' : 0, 'page2' : 0, 'page3' : 0}
actionCtrDistr = {'page1' ; (30, 12), 'page2' ; (60, 25), 'page3' ; (80, 10)}
actionSelCountThreshold = 100
eventCount = 0
actionCount = 0


# send event, for example page visit
def sendEvent(intv):
	while True:
		sessionID = uuid.uuid1()
		r.lpush("eventQueue", sessionID)
		eventCount = eventCount + 1;
		if (eventCount % 20 == 0):
			print "generated %d events" % (eventCount)
		intv = intv + random.randrange(-3, 3)
		time.sleep(intv)

# receive action, for example page to be displayed
def receiveAction(intv):		
	while True:
    	data = r.rpop("actionQueue")
        if data is not None:
        	action = data.split(":")[1]
            updateClickRate(action)
		time.sleep(intv)

#update CTR when enough samples are available 
def updateClickRate(action):
	actionSel[action] = actionSel[action] + 1             
	if (actionSel[action] == actionSelCountThreshold):
		distr = actionCtrDistr[action]
		sum = 0
		for i in range(12):
			sum = sum + random.randrange(1, 100)
		r = float(sum - 600) / 100.0
		r = int(r * distr[1] + distr[0])
		actionSel[action] = 0
		ctr = "%s:%d" % (action, r)
		r.lpush("ctrQueue", ctr)            
	            
eventIntv = float(sys.argv[1])

try:
   thread.start_new_thread(sendEvent, (eventIntv, ))
   thread.start_new_thread(receiveAction, (eventIntv / 2 , ))
except:
   print "Error: unable to start thread"
   
while True:
	pass