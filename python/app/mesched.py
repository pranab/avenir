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
		self.start = None
		self.end = None
		
	def __str__(self):
		"""
		content
		"""
		strDesc = "meeting day {} hour {} min {} duration {} start {} end {}".format(
		self.day, self.hour, self.min, self.duration, self.start, self.end) 
		return strDesc

class MeetingScheduleCost(object):
	"""
	optimize with evolutionary search
	"""
	def __init__(self, numMeeting, numPeople):
		"""
		intialize
		"""
		self.logger = None
		self.numMeeting = numMeeting
		self.people = list(map(lambda i : genNameInitial(), range(numPeople)))
		#self.logger("people" + str(self.people))
		#print(self.people)
		
		#create meetings
		self.participants = list()
		self.partMeetigs = dict()
		self.durations = list()
		dsampler = DiscreteRejectSampler(30,120,30,80,100,60,40)
		for m in range(numMeeting):
			participants = selectRandomSubListFromList(self.people, sampleUniform(2, 6))
			#self.logger("participants" + str(participants))
			#print(participants)
			self.participants.append(participants)
			for p in participants:
				appendKeyedList(self.partMeetigs, p, m)
			self.durations.append(dsampler.sample())
		
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
		
		self.solnCount = 0
		self.invalidSonlCount = 0

	def getMeetings(self, args):
		"""
		meeting list
		"""
		meetings = list()
		for i in range(0, 3 * self.numMeeting, 3):
			mtg = Meeting()
			mtg.day = args[i]
			mtg.hour = args[i+1]
			mtg.min = args[i+2]
			mtg.duration = self.durations[int(i/3)]
			mtg.start =  self.getSecInWeek(mtg.day,  mtg.hour, mtg.min)
			mtg.end = mtg.start + mtg.duration * secInMinute
			self.logger.info(mtg)
			meetings.append(mtg)
		return meetings
		
	def getSecInWeek(self, day, hour, min):
		"""
		sec into week
		"""
		return 	(day - 1) * secInDay + hour * secInHour + min * secInMinute
		
	def isValid(self, args):
		"""
		schedule validation
		"""
		self.solnCount += 1
		meetings = self.getMeetings(args)
		#participants  conflict
		conflicted = False
		for i in range(self.numMeeting-1):
			for j in range(i+1, self.numMeeting):
				cp = findIntersection(self.participants[i], self.participants[j])
				if len(cp) > 0:
					self.logger.debug("common attendees {} {} {}".format(i, j, str(cp)))
					r1 = (meetings[i].start, meetings[i].end)
					r2 = (meetings[j].start, meetings[j].end)
					conflicted = isIntvOverlapped(r1, r2)
					self.logger.debug("r1 {} r2{} conflicted {}".format(str(r1), str(r2), conflicted))
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
					bs = self.getSecInWeek(bh[0], bh[1], 0)
					be = bs + bh[2] * secInHour
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
				misOrdered = not isIntvLess(r1, r2)
				if misOrdered:
					break
		self.logger.info("conflicted {}  bhconflicted {}  misOrdered {}".format(conflicted, bhconflicted, misOrdered))	
					
		valid  = not (conflicted or  bhconflicted or misOrdered)
		if not valid:
			self.invalidSonlCount += 1
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
			
			#meeting per day sorted by time
			meetByDay = dict()
			for m in pmeetings:
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
		self.logger.info("done per person cost")	
			
		#overall cost
		cwl = list(map(lambda p : (pcost[p], self.roleWt[p]) , self.partMeetigs.keys()))
		costs = list(map(lambda cw : cw[0], cwl))
		weights = list(map(lambda cw : cw[1], cwl))
		cost = weightedAverage(costs, weights)
		self.logger.info("cost {:.3f}".format(cost))
		return cost
			
if __name__ == "__main__":
	assert len(sys.argv) == 4, "wrong command line args"
	optConfFile = sys.argv[1]
	numMeeting = int(sys.argv[2])
	numPeople = int(sys.argv[3])
	
	schedCost = MeetingScheduleCost(numMeeting, numPeople)
	optimizer = GeneticAlgorithmOptimizer(optConfFile, schedCost)
	schedCost.logger = optimizer.logger
	optimizer.run()
	print("best soln found")
	print(optimizer.getBest())
	if optimizer.trackingOn:
		print("soln history")
		print(str(optimizer.tracker))
	print("soln count {}  invalid soln count".format(schedCost.solnCount, schedCost.invalidSonlCount))
	
			
