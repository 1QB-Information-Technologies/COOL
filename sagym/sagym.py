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

from abc import abstractmethod
from collections import deque
from gym import spaces
import time
import h5py
import logging
import numpy as np
import threading
import gym
import pdb
import os
import warnings


from sagym.helper import evaluate_success

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

class RenderObjects():
    def __init__(self):
        self.first_render = True

class SAGym(gym.Env):
    def __init__(self):
        self._done = False

        self.max_ep_length = None
        self.set_num_sweeps(100)
        self.action_scaling = 1.0
        self.DESTRUCTIVE_OBSERVATION = False

        self._dump_dataframes = False
        self._episode_counter = 0
        self._last_state = None
        self._step_counter = 0
        self._RO = RenderObjects()
        self._RO.reward_range = [0, 0]
        self._RO.step_range = [0, 0]
        self._RO.episode_reward = []
        self._RO.steps = 0
        self._RO.states = []
        self._RO.Tdat = []
        self._RO.Mdat = []
        self._RO.Edat = []
        self._RO.Adat = []
        self._make_movie = False
        self._RO.success_prob = []
        self.r_deque = deque(maxlen=100)
        self.success_count = deque(maxlen=100)
        self.starttime = time.time()

    @property
    def success_reward(self):
        return 2.5

    def beta_init_function(self):
        return 0.5

    @property
    def _beta_upon_reset(self):
        return self.beta_init_function()

    def set_max_ep_length(self, max_ep_length):
        self.max_ep_length = max_ep_length

    @property
    def lattice(self):
        return self._sa.get_lattice()

    @property
    def lattice2d(self):
        return np.reshape(self.lattice,(-1, self.SPIN_N), order='C')

    @property
    def lattice1d(self):
        return np.reshape(self.lattice,(-1,), order='C')

    @abstractmethod
    def step(self, action):
        """ Override this with a proper action function in child
            classes, but make sure to call this function still with 
            super(...).step()
        """
        self._step_counter += 1
        self._RO.steps += 1
        self._RO.states.append(self._last_state[0])
        self._RO.Tdat.append(1./self._last_state[1])
        self._RO.Mdat.append(self._last_state[2])
        self._RO.Edat.append(self._last_state[3])
        self._RO.Adat.append(np.asscalar(action))
        self._RO.arat.append(self._sa.get_acceptance_ratio())

    def _get_state(self, action=None):
        beta = self._sa.get_current_beta()
        if np.isnan(beta) or beta==0.0:
            beta = self._beta_upon_reset


        _return_2d = True

        if _return_2d:
            s = self.lattice2d
        else:
            s = self.lattice1d
#        s = self.rectifylattice(s)
        self._last_state = [
                s,
                self._sa.get_current_beta(),
                self._sa.get_average_absolute_magnetization(),
                self._sa.get_average_energy()]
        
        return s


    def set_action_scaling(self, action_scaling):
        self.action_scaling = action_scaling

    def toggle_datadump_on(self):
        self._dump_dataframes = True

    def toggle_datadump_off(self):
        self._dump_dataframes = False


    def reset(self):
        self._RO.first_render=True
        self._RO.steps = 0
        self._RO.Tdat = []
        self._RO.Mdat = []
        self._RO.Edat = []
        self._RO.Adat = []
        self._RO.states = []
        self._RO.arat = []
        self._step_counter = 0
        self._last_hard_reset_beta=self._beta_upon_reset
        self._sa.reset(beta=self._last_hard_reset_beta)
        self._energies_before_action = -self._sa.get_all_energies() / self.SPIN_N
        return self._get_state()

    def set_num_sweeps(self, N):
        """
        Set the number of sweeps associated with an action
        """
        self._Nsweeps = N

    def set_beta(self, beta):
        """
        Set the reciprocal temperature
        """
        self._sa.set_current_beta(beta=beta)

    
    def render_close(self):
        pass


class Placeholder(object):
    def __init__(self):
        self.placeholder = True
        pass
    def record(self,*args, **kwargs):
        pass
    def print_to_screen(self):
        pass
    def write(self):
        logging.info("This is a placeholder. Nothing to write")








class SAGymContinuousRandomJ(SAGym):
    def __init__(self):
        super().__init__()

        from sagym.helper import generate_latinit
        self.generate_latinit = generate_latinit

        self.HSR = Placeholder()
        self._first_reset = True
        self.HG = None
        self.experiment_tag = None

        self.graph_model='spin'

        self.SPIN_N = int(os.environ['LATTICE_L'])**2

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(64, self.SPIN_N, 1))

    def initialization_checks(self):
        if self.HG is None:
            raise Exception("You must initialize a Hamiltonian Getter with a call to init_HamiltonianGetter(phase='TEST|TRAIN', [directory=string]) before calling reset()")

        if self.experiment_tag is None:
            raise Exception("You must set an experiment tag with a call to set_experiment_tag(string) before calling reset()")

        if self.max_ep_length is None:
            raise Exception("You must set the maximum number of steps per episode with a call to set_max_ep_length(int)")

    @property
    def success_reward(self):
        return -self.HG.ground_state

    def set_experiment_tag(self, tag):
        self.experiment_tag = tag
        self.results_dir = os.path.join('./results', self.experiment_tag)
        os.makedirs(self.results_dir, exist_ok=True)

    def init_HamiltonianGetter(self, phase='TRAIN', directory=None): 
        if phase=='TRAIN': 
            L = int(os.environ['LATTICE_L']) 
            from sagym.helper import RandomHamiltonianGetter 
            self.HG = RandomHamiltonianGetter(L) 
            from sagym.helper import generate_latinit 
            self.generate_latinit = generate_latinit 
        elif phase=='WSC': 
            from sagym.helper import FileHamiltonianGetter 
            assert directory is not None, "Directory cannot be None" 
            self.HG = FileHamiltonianGetter(directory=directory, disable_random=True, static=0)
            from sagym.helper import generate_WSC_latinit 
            self.generate_latinit = generate_WSC_latinit 
        elif phase=='VALUE_ANALYSIS' or phase=='ISING':
            from sagym.helper import generate_latinit
            self.generate_latinit = generate_latinit
            from sagym.helper import FileHamiltonianGetter 
            assert directory is not None, "Directory cannot be None" 
            self.HG = FileHamiltonianGetter(directory=directory, disable_random=True, static=0) 
        elif phase=='TEST': 
            from sagym.helper import generate_latinit
            self.generate_latinit = generate_latinit
            from sagym.helper import FileHamiltonianGetter 
            assert directory is not None, "Directory cannot be None" 
            self.HG = FileHamiltonianGetter(directory=directory, disable_random=False, static=None) 
        self.HG.get()

        from sagym.sa import SA
        self._sa = SA()


    def init_HamiltonianSuccessRecorder(self, num_hamiltonians, num_trials):
        from sagym.helper import HamiltonianSuccessRecorder
        self.HSR = HamiltonianSuccessRecorder(num_hamiltonians=num_hamiltonians, num_trials=num_trials)
        self.disable_random_Hamiltonians()
        self.truncate_dataset(N=num_hamiltonians)



    def print_hamiltonian_success_recorder_success(self):
        self.HSR.print_success()

    def hsr_write(self):
        self.HSR.write(os.path.join(self.results_dir, 'HamiltonianSuccess.dat'))

    def close_env(self):
        self.hsr_write()

    def set_destructive_observation_on(self):
        self.DESTRUCTIVE_OBSERVATION = True

    def set_static_Hamiltonian_by_ID(self, ID):
        self.HG._static = ID

    def disable_random_Hamiltonians(self):
        self.HG.disable_random = True

    def truncate_dataset(self, N):
        self.HG.truncate_dataset(N)

    #expose some interfaces to get properties of the methods:

    def get_SPIN_N(self):
        return self.SPIN_N

    def get_success_reward(self):
        return self.success_reward

    def get_all_energies(self):
        return self._sa.get_all_energies()

    ####

    def reset(self): 

        if not self._first_reset and not(self.HSR.placeholder):
            self.HSR.record(result=-self._sa.get_all_energies() / self.SPIN_N, goal=self.success_reward, source_dir=self.HG._last_returned_directory)
            self.HSR.print_to_screen()
        else:
            self.initialization_checks()
        
        
        if not(self._first_reset) and self._dump_dataframes:
            os.makedirs(os.path.join(self.results_dir, "dataframes/"),exist_ok=True)
            with h5py.File(os.path.join(self.results_dir,
                                        "dataframes/",
                                        f"episode_{str(self._episode_counter).zfill(6)}.h5"), 'w') as F:
                F.create_dataset("states", data=np.array(self._RO.states))
                F.create_dataset("Tdat", data=np.array(self._RO.Tdat))
                F.create_dataset("Adat", data=np.array(self._RO.Adat))
                F.create_dataset("arat", data=np.array(self._RO.arat))
                F.create_dataset("Mdat", data=np.array(self._RO.Mdat))
                F.create_dataset("Edat", data=np.array(self._RO.Edat))
                success = evaluate_success(-self.get_all_energies() / self.SPIN_N, self.success_reward)
                F.create_dataset("success", data=success)
                F.create_dataset("terminal_energies", data=self.get_all_energies() / self.SPIN_N)
                F.create_dataset("ground_state_energy", data=-self.success_reward)

        self._first_reset = False
        self._episode_counter +=1

        #Call HG.get, which copies a random latfile from a library of latfiles, and returns the ground state energy
        #Generate a random starting configuration

        self.generate_latinit(lines=self.SPIN_N)
        self.sum_of_rewards = 0
        self.HG.get()
        s = super().reset()

        self._actions_taken_since_reset = []
        self.minE_value = 1e8
        self.minE_time = 1

        return self.state_concat(s)

    def state_concat(self, s):
        """This function can be overridden to concatenate
        some internally stored data to the state, e.g. a 
        plane of current temperature, energy, etc."""
        return np.expand_dims(s, axis=-1)

    def step(self, action):  #RANDOM J
        action = action/self.action_scaling
        dbeta = np.asscalar(action)

        penalize_action = False
        if self._sa.get_current_beta() + dbeta <= 0.0001: 
            penalize_action = True
        if self._sa.get_current_beta() + dbeta > 20.0:
            penalize_action = True
            dbeta = 0.0 

        super().step(action)

        if self.DESTRUCTIVE_OBSERVATION:
            # Destroy the system by:
            #   (a) creating a new latinit, and,
            #   (b) resetting the spins.
            self.generate_latinit(lines=self.SPIN_N)
            self._sa.reset(beta=self._last_hard_reset_beta)
            # Then evolve the "new" system by the old policy by
            # repeating all actions that were taking up to this time
            for i,db in enumerate(self._actions_taken_since_reset):
                self._sa.run(self._Nsweeps, float(db))

        self._actions_taken_since_reset.append(dbeta)
        self._sa.run(self._Nsweeps, float(dbeta))

        state = self._get_state(action=action)

        Es = self._sa.get_all_energies()/self.SPIN_N

        if self._step_counter >= self.max_ep_length:
            reward = -np.min(Es) #min energy at final step of episode
            done = True
        else:
            if penalize_action:
                reward = -0.1
            else:
                reward=0
            done = False

        info = {}

        if done:
            self.r_deque.append(reward)
            walltime = time.time() - self.starttime
        self.sum_of_rewards += reward
        
        return self.state_concat(state), reward, done, info

