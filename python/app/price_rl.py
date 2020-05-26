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
import gym
from gym.spaces import Discrete, Box
import numpy as np
import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

"""
Price optimization with Deep reinforcement learning using DQN lgorithm. Code is from the following 
excellent blog with enhancements to make it more usable.

https://blog.griddynamics.com/deep-reinforcement-learning-for-supply-chain-and-price-optimization/
"""

T = 20
price_max = 500
price_step = 10
q_0 = 5000
k = 20
unit_cost = 100
a_q = 300
b_q = 100

price_grid = np.arange(price_step, price_max, price_step)


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
	
	"""
	return np.sqrt(x)
  
def env_intial_state():
	"""
	initial state
	"""
	return np.repeat(0, 2*T)


def q_t(p_t, p_t_1, q_0, k, a, b):
	"""
	Demand at time step t for current price p_t and previous price p_t_1
	"""
	return plus(q_0 - k*p_t - a*shock(plus(p_t - p_t_1)) + b*shock(minus(p_t - p_t_1)))


def profit_t(p_t, p_t_1, q_0, k, a, b, unit_cost):
	"""
	Profit at time step t
	"""
	return q_t(p_t, p_t_1, q_0, k, a, b)*(p_t - unit_cost) 

def profit_t_response(p_t, p_t_1):
	"""
	partial bindings for readability
	"""
	return profit_t(p_t, p_t_1, q_0, k, a_q, b_q, unit_cost)
  
def env_step(t, state, action):
	"""
	next step
	"""
	next_state = np.repeat(0, len(state))
	next_state[0] = price_grid[action]
	next_state[1:T] = state[0:T-1]
	next_state[T+t] = 1
	reward = profit_t_response(next_state[0], next_state[1])
	return next_state, reward
  
def rand_state():
	"""
	random state
	"""
	state = np.repeat(0, 2 * T)
	t = random.randint(5, T - 1)
	for i in range(t):
		pi = random.randint(0, len(price_grid) - 1)
		state[i] = price_grid[pi]
	state[T + t] = 1
	return state
	
class HiLoPricingEnv(gym.Env):
	"""
	pricing environment
	"""
	count = 0
	def __init__(self, config):
		self.t = 0
		self.reset()
		self.action_space = Discrete(len(price_grid))
		self.observation_space = Box(0, 10000, shape=(2*T, ), dtype=np.float32)
		print("observation space  " + str(self.observation_space.shape))
		print("action space  " + str(self.action_space.n))
	
	def reset(self):
		print("** reset " + str(HiLoPricingEnv.count))
		HiLoPricingEnv.count += 1
		self.state = env_intial_state()
		self.t = 0
		return self.state
		
	def step(self, action):
		next_state, reward = env_step(self.t, self.state, action)
		self.t += 1
		self.state = next_state
		return next_state, reward, self.t == T - 1, {}

def create_config():
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
	return config
	
def train_dqn(num_iter):
	"""
	train
	"""
	ray.shutdown()
	ray.init()
	config = create_config()
	trainer = dqn.DQNTrainer(config=config, env=HiLoPricingEnv)
	for i in range(num_iter):
		print("**** next iteration " + str(i))
		HiLoPricingEnv.count = 0
		result = trainer.train()
		print(pretty_print(result))

	policy = trainer.get_policy()
	weights = policy.get_weights()
	#print("policy weights")
	#print(weights)
	
	model = policy.model
	#summary = model.base_model.summary()
	#print("model summary")
	#print(weights)
	
	return trainer

def load_trainer(path):
	"""
	load trainer from checkpoint
	"""
	ray.shutdown()
	ray.init()
	config = create_config()
	trainer = dqn.DQNTrainer(config=config, env=HiLoPricingEnv)
	trainer.restore(path)
	return trainer

def train_incr(path, num_iter):
	"""
	load trainer from checkpoint and incremental training
	"""
	trainer = load_trainer(path)
	for i in range(num_iter):
		print("**** next iteration " + str(i))
		HiLoPricingEnv.count = 0
		result = trainer.train()
		print(pretty_print(result))
	return trainer
	
def get_action(trainer):
	"""
	get action given state from trained model
	"""
	policy = trainer.get_policy()
	state = rand_state()
	print("state:")
	print(state)
	action = policy.compute_single_action(state)
	return action

op = sys.argv[1]
if op == "train":
	"""
	train
	"""
	num_iter = int(sys.argv[2])
	cp_dir = sys.argv[3] if len(sys.argv) == 4 else None
	trainer = train_dqn(num_iter)
	if cp_dir:
		checkpoint = trainer.save(cp_dir)
		print("checkpoint " + checkpoint)

if op == "inctr":
	"""
	train
	"""
	path = sys.argv[2]
	num_iter = int(sys.argv[3])
	cp_dir = sys.argv[4] if len(sys.argv) == 5 else None
	trainer = train_incr(path, num_iter)
	if cp_dir:
		checkpoint = trainer.save(cp_dir)
		print("checkpoint " + checkpoint)
	
elif op == "tract":
	"""
	train and get action
	"""
	num_iter = int(sys.argv[2])
	trainer = train_dqn(num_iter)
	action = get_action(trainer)
	print("action")
	print(action)
	
elif op == "loact":
	"""
	load trainer and get action
	"""
	path = sys.argv[2]
	trainer = load_trainer(path)
	policy = trainer.get_policy()
	state = rand_state()
	print("state:")
	print(state)
	result = policy.compute_single_action(state)
	print("action:")
	print(result)
	
	
elif op == "crstate":
	"""
	create random state
	"""
	state = rand_state()
	print(state)
	
else:
	raise ValueError("invalid command")
	