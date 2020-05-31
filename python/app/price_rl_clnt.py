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

"""
Interactive and online price optimization with DQN
To try this out, in two separate shells run:
    $ ./price_rl_srv.py
    $ ./price_rl_clnt.py --inference-mode=local|remote
    
Adopted from
https://github.com/ray-project/ray/tree/master/rllib/examples/serving
"""

import argparse
import gym

from ray.rllib.env.policy_client import PolicyClient

parser = argparse.ArgumentParser()
parser.add_argument("--no-train", action="store_true", help="Whether to disable training.")
parser.add_argument("--inference-mode", type=str, required=True, choices=["local", "remote"])
parser.add_argument("--off-policy", action="store_true", help="Whether to take random instead of on-policy actions.")
parser.add_argument("--stop-at-reward", type=int, default=9999, help="Stop once the specified reward is reached.")
parser.add_argument("--num-episodes", type=int, default=10, help="Stop once the specified num of episodes are completed")

if __name__ == "__main__":
    args = parser.parse_args()
    env = gym.make(HiLoPricingEnv)
    client = PolicyClient("http://localhost:9900", inference_mode=args.inference_mode)

    eid = client.start_episode(training_enabled=not args.no_train)
    obs = env.reset()
    rewards = 0
	epCount = 0
    while True:
    	#next action
        if args.off_policy:
            action = env.action_space.sample()
            client.log_action(eid, obs, action)
        else:
            action = client.get_action(eid, obs)
            
        # take action and log reward
        obs, reward, done, info = env.step(action)
        rewards += reward
        client.log_returns(eid, reward, info=info)
        
        if done:
            print("Total reward:", rewards)
            if rewards >= args.stop_at_reward:
                print("target reward achieved, exiting")
                exit(0)
            epCount += 1
            if epCount == args.num_episodes:
                print("completed all episodes, exiting")
                exit(0)
               
            #next episode
            rewards = 0
            client.end_episode(eid, obs)
            obs = env.reset()
            eid = client.start_episode(training_enabled=not args.no_train)
            