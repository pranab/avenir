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
Interactive and online price optimization
To try this out, in two separate shells run:
    $ ./price_rl_srv.py
    $ ./price_rl_clnt.py --inference-mode=local|remote
    
Adopted from
https://github.com/ray-project/ray/tree/master/rllib/examples/serving
"""

import argparse
import os

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.logger import pretty_print

SERVER_ADDRESS = "localhost"
SERVER_PORT = 9900
CHECKPOINT_FILE = "price_checkpoint_loc.out"

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="DQN")
parser.add_argument("--framework", type=str, choices=["tf", "torch"], default="tf")
parser.add_argument("--modDir", type=str, default="./model/price")

if __name__ == "__main__":
	args = parser.parse_args()
    ray.init()

    env = HiLoPricingEnv
    connector_config = {
        # Use the connector server to generate experiences.
        "input": (
            lambda ioctx: PolicyServerInput( \
                ioctx, SERVER_ADDRESS, SERVER_PORT)
        ),
        # Use a single worker process to run the server.
        "num_workers": 0,
        # Disable OPE, since the rollouts are coming from online clients.
        "input_evaluation": [],
    }

    if args.run == "DQN":
        # Example of using DQN (supports off-policy actions).
        trainer = DQNTrainer(
            env=env,
            config=dict(
                connector_config, **{
                    "exploration_config": {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 1.0,
                        "final_epsilon": 0.02,
                        "epsilon_timesteps": 1000,
                    },
                    "learning_starts": 100,
                    "timesteps_per_iteration": 200,
                    "log_level": "INFO",
                    "framework": args.framework,
                }))
    elif args.run == "PPO":
        # Example of using PPO (does NOT support off-policy actions).
        trainer = PPOTrainer(
            env=env,
            config=dict(
                connector_config, **{
                    "sample_batch_size": 1000,
                    "train_batch_size": 4000,
                    "framework": args.framework,
                }))
    else:
        raise ValueError("--run must be DQN or PPO")

	#this file contains checkpoint file path
    checkpoint_path_file = CHECKPOINT_FILE

    # Attempt to restore from checkpoint if possible.
    if os.path.exists(checkpoint_path_file):
        checkpoint_path = open(checkpoint_path_file).read()
        if os.path.exists(checkpoint_path):
        	print("Restoring from checkpoint path", checkpoint_path)
        	trainer.restore(checkpoint_path)
        else:
        	print("checkpoint file does not exist")
	else:
		print("file containing checkpoint file path does not exist")
		
    # Serving and training loop
    while True:
        print(pretty_print(trainer.train()))
        checkpoint = trainer.save(parser.modDir)
        print("Last checkpoint", checkpoint)
        with open(checkpoint_path, "w") as f:
            f.write(checkpoint)
            
