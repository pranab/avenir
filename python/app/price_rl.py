#!/usr/local/bin/python3
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

import os
import sys
import random
import numpy as np
import math
import gym
from gym.spaces import Discrete, Box
import numpy as np
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print
sys.path.append(os.path.abspath("../lib"))
from util import *


"""
Price optimization with Deep reinforcement learning using DQN lgorithm. Code is from the following 
excellent blog with enhancements to make it more usable.

https://blog.griddynamics.com/deep-reinforcement-learning-for-supply-chain-and-price-optimization/
"""

T = 20					# time steps in the price schedule
priceMin = 400			# minimum valid price
priceMax = 500			# maximum valid price
priceStep = 5			# price schedule step
demandIntcpt = 5000		# Intercept in the demand function 
k1 = -5.0				# first order coeff in the demand function
k2 = -0.1				# second order coeff in the demand function,
unitCost = 100			# product production cost,
aPrInc = 300			# response coefficient for price increase
bPrDec = 100			# response coefficient for price decrease
demCycPer = 5 * T		# cyclic demand period
demCycAmp = 500			# cyclic demand amp
ss = 2 * T + 1			# state space size 0 to T-1 past prices, T to 2T -1 one hot vector for current price, 2T cycle offset
randDemSd = 100			# std dev of random demand

priceGrid = np.arange(priceMin, priceMax, priceStep)


def plus(x):
	"""
	Environment simulator
	"""
	return 0 if x < 0 else x

def minus(x):
	"""
	force negative
	"""
	return 0 if x > 0 else -x

def shock(x):
	"""
	non linear
	"""
	return np.sqrt(x)
  
def envIntialState():
	"""
	initial state
	"""
	st =  np.repeat(0, ss)
	cycOff = random.randint(0, demCycPer)
	st[ss - 1] = cycOff
	return st


def curDemand(curPrice, prevPrice, demandIntcpt, k1, k2, a, b, coff):
	"""
	demand at time step t for current price curPrice and previous price prevPrice
	"""
	# price dependent
	pdm = curPrice - priceMin
	pdp = curPrice - prevPrice
	q =  demandIntcpt + k1 * pdm + k2 * pdm * pdm  - a * shock(plus(pdp)) + b * shock(minus(pdp)) 
	
	# cyclic 
	q += demCycAmp * math.sin(2.0 * math.pi * coff / demCycPer)
	
	# random
	q += np.random.normal(0, randDemSd)
	
	q =  plus(q)
	return q


def curProfit(curPrice, prevPrice, demandIntcpt, k1, k2, a, b, unitCost, coff):
	"""
	Profit at time step t
	"""
	return curDemand(curPrice, prevPrice, demandIntcpt, k1, k2, a, b, coff) * (curPrice - unitCost) 

def curProfitResponse(curPrice, prevPrice, coff):
	"""
	partial bindings for readability
	"""
	return curProfit(curPrice, prevPrice, demandIntcpt, k1, k2, aPrInc, bPrDec, unitCost, coff)
  
def envStep(t, state, action):
	"""
	next step
	"""
	nextState = np.repeat(0, len(state))
	
	#price history
	nextState[0] = priceGrid[action]
	nextState[1:T] = state[0:T-1]
	
	#cuurent price one hot vec
	nextState[T+t] = 1
	
	#cycle offset
	nextState[ss - 1] = (state[ss - 1] + 1) % demCycPer
	
	reward = curProfitResponse(nextState[0], nextState[1], nextState[ss - 1])
	return nextState, reward
  
def randState():
	"""
	random state
	"""
	state = np.repeat(0, ss)
	t = random.randint(5, T - 1)
	for i in range(t):
		pi = random.randint(0, len(priceGrid) - 1)
		state[i] = priceGrid[pi]
	state[T + t] = 1
	state[ss - 1] = random.randint(0, demCycPer)
	return state
	
class HiLoPricingEnv(gym.Env):
	"""
	pricing environment
	"""
	count = 0
	def __init__(self, config):
		self.t = 0
		self.reset()
		self.action_space = Discrete(len(priceGrid))
		self.observation_space = Box(0, 10000, shape=(ss, ), dtype=np.float32)
		print("observation space  " + str(self.observation_space.shape))
		print("action space  " + str(self.action_space.n))
	
	def reset(self):
		HiLoPricingEnv.count += 1
		self.state = envIntialState()
		self.t = 0
		return self.state
		
	def step(self, action):
		nextState, reward = envStep(self.t, self.state, action)
		self.t += 1
		self.state = nextState
		return nextState, reward, self.t == T - 1, {}

def createConfig():
	"""
	configuration
	"""
	config = dqn.DEFAULT_CONFIG.copy()
	config["log_level"] = "WARN"
	config["lr"] = 0.002
	config["gamma"] = 0.80
	config["train_batch_size"] = 256
	config["buffer_size"] = 10000
	config["timesteps_per_iteration"] = 5000
	config["hiddens"] = [128, 128, 128]
	config["exploration_final_eps"] = 0.01

	#config["use_exec_api"] = False
	return config
	
def trainDqn(numIter):
	"""
	train
	"""
	ray.shutdown()
	ray.init()
	config = createConfig()
	trainer = dqn.DQNTrainer(config=config, env=HiLoPricingEnv)
	for i in range(numIter):
		print("\n**** next iteration " + str(i))
		HiLoPricingEnv.count = 0
		result = trainer.train()
		print(pretty_print(result))
		print("env reset count " + str(HiLoPricingEnv.count))

	policy = trainer.get_policy()
	weights = policy.get_weights()
	#print("policy weights")
	#print(weights)
	
	model = policy.model
	#summary = model.base_model.summary()
	#print("model summary")
	#print(weights)
	
	return trainer

def loadTrainer(path):
	"""
	load trainer from checkpoint
	"""
	ray.shutdown()
	ray.init()
	config = createConfig()
	trainer = dqn.DQNTrainer(config=config, env=HiLoPricingEnv)
	trainer.restore(path)
	return trainer

def trainIncr(path, numIter):
	"""
	load trainer from checkpoint and incremental training
	"""
	trainer = loadTrainer(path)
	for i in range(numIter):
		print("\n**** next iteration " + str(i))
		HiLoPricingEnv.count = 0
		result = trainer.train()
		print(pretty_print(result))
		print("env reset count " + str(HiLoPricingEnv.count))
	return trainer
	
def getAction(trainer):
	"""
	get action given state from trained model
	"""
	policy = trainer.get_policy()
	state = randState()
	print("state:")
	print(state)
	action = policy.compute_single_action(state)
	return action

def validateCpDir(cpDir):
	"""
	checks if checkpoint dir exists
	"""
	if cpDir and not os.path.isdir(cpDir):
		raise ValueError("provided checkpoint directory does not exist")

def validateCpFile(cpFile):
	"""
	checks if checkpoint file exists
	"""
	if not os.path.isfile(cpFile):
		raise ValueError("provided checkpoint file does not exist")

if __name__ == "__main__":
	op = sys.argv[1]
	if op == "train":
		"""
		train
		"""
		print("******** training and optionally checkpointing ********")
		numIter = int(sys.argv[2])
		cpDir = sys.argv[3] if len(sys.argv) == 4 else None
		validateCpDir(cpDir)
		trainer = trainDqn(numIter)
		if cpDir:
			checkpoint = trainer.save(cpDir)
			print("checkpoint " + checkpoint)

	elif op == "inctr":
		"""
		train
		"""
		print("******** incremental training on checkpointed model and then optionally checkpoint again ********")
		path = sys.argv[2]
		validateCpFile(path)
		numIter = int(sys.argv[3])
		cpDir = sys.argv[4] if len(sys.argv) == 5 else None
		validateCpDir(cpDir)
		trainer = trainIncr(path, numIter)
		if cpDir:
			checkpoint = trainer.save(cpDir)
			print("checkpoint " + checkpoint)
	
	elif op == "tract":
		"""
		train and get action
		"""
		print("******** training without checkpointing and then getting action ********")
		numIter = int(sys.argv[2])
		trainer = trainDqn(numIter)
		action = getAction(trainer)
		print("action")
		print(action)
	
	elif op == "loact":
		"""
		load trainer and get action
		"""
		print("******** loading checkpointed model and getting action ********")
		path = sys.argv[2]
		validateCpFile(path)
		state = sys.argv[3] if len(sys.argv) == 4 else None
		if state:
			#provided state
			state = toIntList(state.split(","))
			assert len(state) == ss, "invalid state size"
		else:
			# random state
			print("creating random but valid state")
			state = randState()
		trainer = loadTrainer(path)
		policy = trainer.get_policy()
		print("state:")
		print(state)
		result = policy.compute_single_action(state)
		print("action:")
		print(result)
	
	
	elif op == "crstate":
		"""
		create random state
		"""
		state = randState()
		print(state)
	
	else:
		raise ValueError("invalid command")
	