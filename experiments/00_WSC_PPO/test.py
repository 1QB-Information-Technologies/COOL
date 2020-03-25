"""-----------------------------------------------------------------------------

Copyright (C) 2019-2020 1QBit
Contact info: Pooya Ronagh <pooya@1qbit.com>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------------"""

import gym
import sagym
import numpy as np
import time
from tqdm import tqdm
from policies import CnnPolicyOverReps
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
from stable_baselines import PPO2
import os
import sys
from train import model_args, mostrecentmodification, env_generator
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tag", help="Experiment tag (name) used for organizing output", default="train")
parser.add_argument("-d","--hamiltonian_directory", help="Hamiltonian directory", default="")
parser.add_argument("-b","--betainit", help="Initial inverse temperature", default="")
args = parser.parse_args()


silent = True

from train import episode_length
experiment_name=sys.argv[1]
experiment_description="""Reward is the negative of the minimum energy at episode termination, with no episode termination if negative beta encountered"""

beta_init = float(args.betainit)

env = DummyVecEnv([lambda: env_generator(ep_len=episode_length, total_sweeps=episode_length*100, beta_init_function=lambda: beta_init )])
env = VecNormalize(env, norm_obs=False, norm_reward=False, training=False)

env.env_method('set_experiment_tag', indices=[0], tag=args.tag)
env.env_method('set_max_ep_length', indices=[0], max_ep_length=episode_length)
env.env_method("toggle_datadump_on", indices=[0])
#env.env_method('init_HamiltonianGetter', indices=[0], phase='TEST', directory=args.hamiltonian_directory )
env.env_method('init_HamiltonianGetter', indices=[0], phase='WSC', directory=args.hamiltonian_directory )



#Attempting to restore most recently saved model
print("!!!!!!!!!!!!!!!!!!!!!!!")
max_path = mostrecentmodification(os.path.join("./saves", 'WSC'))#args.tag))
print(f"   Attempting to restore model {max_path}")
model = PPO2.load(max_path, env=env, **model_args)
print("!!!!!!!!!!!!!!!!!!!!!!!")

env = model.get_env()

num_hamiltonians = 1
num_trials = 100
env.env_method("init_HamiltonianSuccessRecorder", indices=[0], num_hamiltonians=num_hamiltonians, num_trials=num_trials)
env.env_method("set_static_Hamiltonian_by_ID", indices=[0], ID=0)
obs = env.reset()

test_ep=-1
for ham in range(num_hamiltonians):
    env.env_method("set_static_Hamiltonian_by_ID", indices=[0], ID=ham)
    for trial in range(num_trials): 
        test_ep += 1
        state = None
        done = [False for _ in range(env.num_envs)]
        step=-1
        while True:
            step+=1

            action, state = model.predict(obs, state=state, mask=done, deterministic=True)
            obs, reward, d, _ = env.step(action)

            if test_ep==1000:
                env.env_method("toggle_datadump_off", indices=[0])

            if d:
                if not silent:
                    TL.plot()
                break
env.env_method('close_env')

print(f"SA_RESULT: {beta_init} 0.0 ", end='')
env.env_method('print_hamiltonian_success_recorder_success', indices=[0])
print("")
