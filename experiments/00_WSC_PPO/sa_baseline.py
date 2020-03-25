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
from policies import CnnPolicyOverReps, CnnLnLstmPolicyOverReps
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
from stable_baselines import PPO2
import os
import sys
from train import model_args, mostrecentmodification, env_generator
from sagym.render.moviemaker import MovieMaker
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tag", help="Experiment tag (name) used for organizing output", default="train")
parser.add_argument("-d","--hamiltonian_directory", help="Hamiltonian directory", default="")
args = parser.parse_args()


silent = True

from train import episode_length
experiment_name=sys.argv[1]
experiment_description=""" """


beta_init=float(args.tag.split('-')[-2])
beta_end= float(args.tag.split('-')[-1])


env = DummyVecEnv([lambda: env_generator(ep_len=episode_length, total_sweeps=episode_length*100, beta_init_function=lambda: beta_init )])
env = VecNormalize(env, norm_obs=False, norm_reward=False, training=False)

env.env_method('set_experiment_tag', indices=[0], tag=args.tag)
env.env_method('set_max_ep_length', indices=[0], max_ep_length=episode_length)
env.env_method("toggle_datadump_on", indices=[0])
env.env_method('init_HamiltonianGetter', indices=[0], phase='WSC', directory=args.hamiltonian_directory )


##This takes a while. If we don't need the policy, comment the next two lines.
#model = PPO2(CnnLnLstmPolicyOverReps, env, verbose=True, **model_args)
#env = model.get_env()

num_hamiltonians = 1
num_trials = 1000
env.env_method("init_HamiltonianSuccessRecorder", indices=[0], num_hamiltonians=num_hamiltonians, num_trials=num_trials)
env.env_method("set_static_Hamiltonian_by_ID", indices=[0], ID=0)
obs = env.reset()


dbeta = (beta_end - beta_init) / episode_length
action = dbeta


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

#            action, state = model.predict(obs, state=state, mask=done, deterministic=True)
            obs, reward, d, _ = env.step(np.array([action]))
#            env.env_method("render", indices=[0], mode='human', destination='x')

            if test_ep==1000:
                env.env_method("toggle_datadump_off", indices=[0])

            if d:
                break
env.env_method('close_env')

print(f"SA_RESULT: {beta_init} {beta_end} ", end='')
env.env_method('print_hamiltonian_success_recorder_success', indices=[0])
print("")

