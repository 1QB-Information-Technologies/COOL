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

import os
import gym
import sys
import sagym
import numpy as np
from policies import CnnPolicyOverReps, CnnLnLstmPolicyOverReps
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from sagym.helper import TTSLogger
import logging
import argparse
import h5py

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tag", help="Experiment tag (name) used for organizing output", default="train")
    parser.add_argument("-d","--hamiltonian_directory", help="Hamiltonian directory", default="")
    parser.add_argument("--destructive", default=False, action='store_true', help='Whether or not to use destructive observation.')
    args = parser.parse_args()

episode_length = 40

model_args = dict(
        gamma=0.99,
        n_steps=episode_length*8,
        ent_coef=0.001,
        learning_rate=1e-6,
        vf_coef=0.9,
        nminibatches=1,
        noptepochs=8,
        cliprange=0.05,
        tensorboard_log='./tensorboard/',
    )

def mostrecentmodification(directory):
    max_mtime = 0
    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        mtime = os.stat(full_path).st_mtime
        print(f,mtime)
        if mtime > max_mtime:
            max_mtime = mtime
            max_path = full_path
    return max_path

log_dir="./logs/"
os.makedirs(log_dir, exist_ok=True)

def env_generator(ep_len=40, total_sweeps=4000, beta_init_function=None):

    env = gym.make('SAContinuousRandomJ-v0')
    env.unwrapped.set_max_ep_length(ep_len)
    env.unwrapped.set_num_sweeps(total_sweeps // ep_len)
    env.unwrapped.action_scaling = 5.0

    if beta_init_function is None:
        env.unwrapped.beta_init_function = lambda: 0.3333
    else:
        env.unwrapped.beta_init_function = beta_init_function
    env = Monitor(env, log_dir, allow_early_resets=True)

    return env


#PHASE='VALUE_ANALYSIS'    # for writing out the value function
#PHASE='ISING'             # for training ISING model ONLY
PHASE='TRAIN'

if PHASE=='VALUE_ANALYSIS':
    model_args["learning_rate"] = 0.0000000000
    model_args["noptepochs"] = 1

if __name__=='__main__':

    env = DummyVecEnv([lambda: env_generator(ep_len=episode_length, total_sweeps=episode_length*1, beta_init_function=lambda: 2*np.random.rand() + 0.333 )])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, training=True)

    env.env_method('set_experiment_tag', indices=[0], tag=args.tag)
    if PHASE=='VALUE_ANALYSIS' or PHASE=='ISING':
        env.env_method('init_HamiltonianGetter', indices=[0], phase=PHASE, directory=args.hamiltonian_directory)
    elif PHASE=='TRAIN':
        env.env_method('init_HamiltonianGetter', indices=[0], phase='TRAIN')
    env.env_method('set_max_ep_length', indices=[0], max_ep_length=episode_length)

    if args.destructive:
        env.env_method('set_destructive_observation_on', indices=[0])


    n_steps = 0
    best_mean_reward = -np.inf

    if PHASE=='VALUE_ANALYSIS':
        value_file = h5py.File(os.path.join("results/", args.tag, "value_function_dump.h5"), 'w')
        first_dump = True


    def callback(_locals, _globals):

        if PHASE=='VALUE_ANALYSIS':
            value_shape = _locals['values'].shape
            observation_shape = _locals['obs'].shape
            actions_shape = _locals['actions'].shape

            global first_dump, value_file
            if first_dump:
                value_file.create_dataset("values", data=_locals['values'], maxshape=(None,))
                value_file.create_dataset("obs", data=_locals['obs'], maxshape=tuple([None,] + list(observation_shape)[1:]))
                value_file.create_dataset("actions", data=_locals['actions'], maxshape=tuple([None,] + list(actions_shape)[1:])  )
                first_dump = False
            else:
                for key in ['values', 'obs', 'actions']:
                    value_file[key].resize(value_file[key].shape[0] + _locals[key].shape[0], axis=0)
                    value_file[key][-_locals[key].shape[0]:] = _locals[key]

            if value_file['values'].shape[0] > 5000000:
                #error out after we have enough data. Hacky, but whatever.
                value_file.close()
                skjhdfksjdh()


        global n_steps, best_mean_reward
        if (n_steps) % 100 == 0:
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                os.makedirs(os.path.join('./saves/',args.tag), exist_ok=True)
                _locals['self'].save(os.path.join('./saves/',args.tag,f"saved_model_{n_steps}"))
        n_steps += 1
        return True



    try:
        model = PPO2.load(mostrecentmodification(os.path.join("./saves/",args.tag)), env=env, **model_args)
    except:
        print("ERROR restoring model. Starting from scratch")
        model = PPO2(CnnLnLstmPolicyOverReps, env, verbose=True, **model_args)

    model.learn(total_timesteps=int(5000000), callback=callback, reset_num_timesteps=False, tb_log_name=args.tag)




