#!/usr/local/bin/python3

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
import jprops
import numpy as np
from statistics import mean
sys.path.append(os.path.abspath("../lib"))
sys.path.append(os.path.abspath("../mlextra"))
from util import *
from optpopu import *
from optsolo import *
from sampler import *

"""
Meeting schedule optimization with genetic algorithm. Maximizes average free time slots
subjects to constraints
"""

class Meeting(object):
	"""
	optimize with evolutionary search
	"""
	def __init__(self):
		"""
		intialize
		"""
		self.day = None
		self.hour = None
		self.min = None
		self.duration = None
		self.start = none
		self.end = None

class MeetingScheduleCost(object):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, numMeeting, numPeople):
		"""
		intialize
		"""
		self.numMeeting = numMeeting
		self.people = list(map(lambda i : genNameInitial(), range(numPeople)))
		
		#create meetings
		self.participants = list()
		self.partMeetigs = dict()
		for m in range(numMeeting):
			particinats = selectRandomSubListFromList(self.people, sampleUniform(2, 6))
			self.participantsappend(particinats)
			for p in particinats:
				appendKeyedList(self.partMeetigs, p, m)
		
		#constrain time ordered meetings
		self.ordMeetings = list()		
		self.ordMeetings.append(selectRandomSubListFromList(np.arange(numMeeting), 2))
		
		#constrain blocked time
		self.blockedHrs = dict()
		pbl = selectRandomSubListFromList(self.people, 2)
		for pb in pbl:
			day = sampleUniform(1, 5)
			hr = sampleUniform(8, 15)
			du = sampleUniform(1, 3)
			blocked = (day, hr, du)
			self.blockedHrs[p] = blocked
			
		#weight as per role
		self.roleWt = dict.fromkeys(self.people, 1.0)
		numMan = int(0.2 * numPeople)
		managers = selectRandomSubListFromList(self.people, numMan)
		for ma in managers:
			self.roleWt[ma] = 1.3
		ma = selectRandomFromList(managers)
		self.roleWt[ma] = 1.8
		
	def getMeetings(self, args):
		"""
		meeting list
		"""
		meetings = list()
		for i in range(0, 4 * self.numMeeting, 4):
			mtg = Meeting()
			mtg.day = args[i]
			mtg.hour = args[i+1]
			mtg.min = args[i+2]
			mtg.duration = args[i+3]
			mtg.start = (mtg.day - 1) * secInDay + mtg.hour * secInHour + mtg.min * secInMinute
			mtg.end = mtg.start + mtg.duration * secInMinute
			meetings.append(mtg)
		return meetings
		
	def isValid(self, args):
		"""
		schedule validation
		"""
		meetings = self.getMeetings(args)
		
		#participants  conflict
		conflicted = False
		for i in range(self.numMeeting):
			for j in range(i, self.numMeeting):
				cp = findIntersection(self.participants[i], self.participants[j])
				if len(cp) > 0:
					r1 = (meetings[i].start, meetings[i].end)
					r1 = (meetings[j].start, meetings[j].end)
					conflicted = isIntvOverlapped(r1, r2)
					if conflicted:
						break
			if conflicted:
				break
						
		#blocked hour conflict
		bhconflicted = False
		for p in self.partMeetigs.keys():
			mids = self.partMeetigs[p]
			pmeetings = list(map(lambda m : meetings[m], mids))
			for m in pmeetings:
				r1 = (m.start, m.end)
				if p in self.blockedHrs:
					bh = self.blockedHrs[p]
					if bh[0] == m.day:
						bs = bh[1] * secInHour
						be = (bh[1] + bh[1]) * secInHour
						r2 = (bs, be)
						bhconflicted = isIntvOverlapped(r1, r2)
						if bhconflicted:
							break
			if bhconflicted:
				break

		#meeting order
		misOrdered = False
		if not (conflicted or bhconflicted):
			for om in self.ordMeetings:
				r1 = (meetings[om[0]].start, meetings[om[0]].end)
				r1 = (meetings[om[1]].start, meetings[om[1]].end)
				misOrdered = not isIntvLess(rOne, rTwo)
				if misOrdered:
					break
					
		valid  = not (conflicted or  bhconflicted or misOrdered)
		return valid

	def evaluate(self, args):
		"""
		cost
		"""
		meetings = self.getMeetings(args)

		#cost for each person
		pcost = dict()
		for p in self.partMeetigs.keys():
			mids = self.partMeetigs[p]
			pmeetings = list(map(lambda m : meetings[m], mids))
			
			#meeting per day sorted by hour
			meetByDay = dict()
			for m in range(pmeetings):
				appendKeyedList(meetByDay, m.day, m)
			for d in meetByDay.keys():
				meetByDay[d].sort(key = lambda m : m.start)
			
			#free slots
			fslots = list()
			for d in meetByDay.keys():
				dmeetings = meetByDay[d]
				pend = 8 * secInHour
				for dm in dmeetings:
					ft = dm.start - pend
					fslots.append(ft)
					pend = dm.end
				ft = 18 * secInHour - dmeetings[-1].end
				fslots.append(ft)
			avFt = mean(fslots)
			cost = 8 - avFt / secInHour
			pcost[p] = cost
				
		#overall cost
		cwl = list(map(lambda p : (pcost[p], self.roleWt[p]) , self.partMeetigs.keys()))
		costs = list(map(lambda cw : cw[0], cwl))
		weights = list(map(lambda cw : cw[1], cwl))
		cost = weightedAverage(costs, weights)
		
		return cost
			
if __name__ == "__main__":
	assert len(sys.argv) == 4, "wrong command line args"
	optConfFile = sys.argv[1]
	numMeeting = int(sys.argv[2])
	numPeople = int(sys.argv[3])
	
	schedCost = MeetingScheduleCost(numMeeting, numPeople)
	optimizer = GeneticAlgorithmOptimizer(optConfFile, schedCost)
	optimizer.run()
	print("best soln found")
	print(optimizer.getBest())
	if optimizer.trackingOn:
		print("soln history")
		print(str(optimizer.tracker))
	
			
