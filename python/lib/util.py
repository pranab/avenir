#!/Users/pranab/Tools/anaconda/bin/python

import sys
from random import randint
import time
import uuid
from datetime import datetime

tokens = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M",
	"N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]

def genID(len):
	id = ""
	for i in range(len):
		id = id + selectRandomFromList(tokens)
	return id
		
		
def selectRandomFromList(list):
	return list[randint(0, len(list)-1)]
	
def selectRandomSubListFromList(list, num):
	sel = selectRandomFromList(list)
	selSet = {sel}
	selList = [sel]
	while (len(selSet) < num):
		sel = selectRandomFromList(list)
		if (sel not in selSet):
			selSet.add(sel)
			selList.append(sel)		
	return selList
	
def genIpAddress():
	i1 = randint(0,256)
	i2 = randint(0,256)
	i3 = randint(0,256)
	i4 = randint(0,256)
	ip = "%d.%d.%d.%d" %(i1,i2,i3,i4)
	return ip
	
def curTimeMs():
	return int((datetime.utcnow() - datetime(1970,1,1)).total_seconds() * 1000)
	
def secDegPolyFit(x1, y1, x2, y2, x3, y3):
	t = (y1 - y2) / (x1 - x2)
	a = t - (y2 - y3) / (x2 - x3)
	a = a / (x1 - x3)
	b = t - a * (x1 + x2)
	c = y1 - a * x1 * x1 - b * x1
	return (a, b, c)

def range_limit(val, min, max):
	if (val < min):
		val = min
	elif (val > max):
		val = max
	return val	
	