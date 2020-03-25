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

import numpy as np
import math
import os
import shutil
import logging
import time


def generate_latinit_bernoulli(lines):
    """Uniformly distributed number of up/down spins"""
    num_up = np.random.randint(0,lines)
    spins = np.array([1]*num_up + [-1]*(lines-num_up)).astype(np.int)
    np.random.shuffle(spins)


    with open("./latinit", "w") as F:
        for i in range(lines):
            pass
            F.write(f"{spins[i]}\n")

def generate_WSC_latinit(lines):
    spins = [-1]*8 + [1.0]*8

    with open("./latinit", "w") as F:
        for i in range(lines):
            F.write(f"{spins[i]}\n")

def generate_latinit_random(lines):
    """Each spin randomly drawn with 50% up/down probability"""
    #num_up = np.random.randint(0,lines)
    #spins = [1.0]*num_up + [-1.0]*(lines-num_up)
    spins = 2*np.random.randint(0,2,lines)-1
    #np.random.shuffle(spins)

    with open("./latinit", "w") as F:
        for i in range(lines):
            F.write(f"{spins[i]}\n")

def generate_latinit(lines):
    generate_latinit_bernoulli(lines)









def evaluate_success(result, goal):
    return float(min(1.0, sum((goal - result) < 1e-5)))



class HamiltonianSuccessRecorder(object):
    def __init__(self, num_hamiltonians, num_trials):
        self.placeholder = False #indicate to the environment this is not a placeholder recorder class and that it is, in fact, real
        self.source_dir = ["empty" for _ in range(num_hamiltonians)]
        self.num_trials = num_trials
        self.num_hamiltonians = num_hamiltonians
        self.counter = 0
        self.all_results = np.zeros(self.num_trials*self.num_hamiltonians) + np.nan
        self.result_dict = dict()


    def record(self, result, goal, source_dir='NA'):
        print("")
        print("----- HamiltonianSuccessRecorder -----")
        print(f"  Goal: {goal}")
        print(f"  H(all):", result)
        success = int(evaluate_success(result, goal))
        print(f"-{'success' if success else 'fail---'}------------------------------")
        self.all_results[self.counter] = success
        self.counter += 1
        if source_dir not in self.result_dict:
            self.result_dict[source_dir] = [success,]
        else:
            self.result_dict[source_dir].append(success)
        

    def print_to_screen(self):
        print("")
        for i,h in enumerate(self.result_dict):
            mean = np.mean(self.result_dict[h])*100.
#            print(f"Instance {i:5d}:  {mean:5.2f}  {h.split('/')[-1]}")
            print(f"Instance {i:5d}:  {self.result_dict[h]}")
        print(f"Overall: {np.nanmean(self.all_results)*100:5.3f}%")


    def write(self, outputfile):
        with open(outputfile, 'w') as F:
            #F.write("Instance, Hamiltonian, Trial successes... \n")
            for i,h in enumerate(self.result_dict):
                #F.write(f"{int(i):5d} {h.split('/')[-1]} ")
                for success in self.result_dict[h]:
                    F.write(f"{int(success):3d},")
                #F.write("\n")


    def print_success(self):
        print(np.nanmean(self.all_results)*100)


#from sagym.models import make_rndj_nn_sg as make_latfile
from sagym.models import make_rndj_notrunc_nn_sg as make_latfile
class RandomHamiltonianGetter(object):
    def __init__(self, L):
        self.L = L
    
    def get(self):
        make_latfile('./latfile', L=self.L)
    
    @property
    def ground_state(self):
        raise Exception("You may not request the ground state of a RandomHamiltonian.  This is prohibited as the algorithm must work without knowledge of the ground state")

    @property
    def ground_state_config(self):
        raise Exception("You may not request the ground state of a RandomHamiltonian.  This is prohibited as the algorithm must work without knowledge of the ground state")

    def report_energy(self, energy, state=None):
        """ This is here to play nicely, but it should do nothing. """
        pass

class FileHamiltonianGetter(object):
    def __init__(self, directory, disable_random, static=None):
        self._disable_random=disable_random
        self._last_idx = 0
        self._directory = directory
        self._list_dir = sorted(os.listdir(self._directory))
        self._last_returned_directory = None
        self._static = static
        print(f"{len(self._list_dir)} directories found")

    @property
    def ground_state(self):
        return self._last_returned_gs_energy

    def truncate_dataset(self, N):
        logging.info(f"Truncating list of Hamiltonians to {N} elements")
        logging.info(f"--> original list size: {len(self._list_dir)}")
        self._list_dir = self._list_dir[0:N]
        logging.info(f"--> new list size: {len(self._list_dir)}")

    def get(self):
        dirr = self._list_dir[self._last_idx % len(self._list_dir)]
        self._last_idx += 1
        if not self._disable_random:
            dirr = np.random.choice(self._list_dir)

        #if static, return the same Hamiltonian every time:
        if self._static is not None:
            dirr = self._list_dir[self._static % len(self._list_dir)]

        self._last_returned_directory = os.path.join(self._directory, dirr)
        shutil.copyfile(self._last_returned_directory + '/latfile', "./latfile")
        try:
            self._last_returned_gs_energy = float(open(self._directory + dirr + "/gs_energy", 'r').read())
        except:
            logging.error(f"Could not open gs_energy sidecar file for {self._last_returned_directory}.  Please make sure the reference energy is present.")
            raise Exception

        return self._last_returned_gs_energy

    def report_energy(self, energy):
        return
#        if (energy-self._last_returned_gs_energy) < -1e-6:
#            logging.debug(f"Found lower ground state energy ({energy} vs {self._last_returned_gs_energy}). Updating sidecar file {self._last_returned_directory}")
#            with open(self._last_returned_directory + "/gs_energy", 'w') as F:
#                F.write(f"{energy:.6f}")
#            self._last_returned_gs_energy = energy


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())



class TTSLogger(object):
    def __init__(self, max_time, time_bins):

        self.sum = np.zeros(time_bins, dtype=np.float32)
        self.sum2 = np.zeros(time_bins, dtype=np.float32)
        self.n = np.zeros(time_bins, dtype=np.int)
        self.running_stats = [RunningStats() for _ in range(time_bins)]
        self.max_time = max_time
        self.times = np.linspace(0, max_time, num=time_bins)
        self._fig, self._ax = plt.subplots(1,1,figsize=(6,6))
        self._first_plot = True

    def record_frame(self, result, goal, time):
        """result is an array of results.  If any element of result is equal to goal, this is counted as a success"""
        bin_idx = int(np.floor(time / (self.max_time / len(self.sum))))
        success = evaluate_success(result, goal)
        self.sum[bin_idx] += success
        self.sum2[bin_idx] += success*success
        self.n[bin_idx] += 1

        self.running_stats[bin_idx].push(success)
        #self.success[bin_idx] = self.success[bin_idx] + (success - self.success[bin_idx])/(self.n[bin_idx] + 1)

    @property
    def sd(self):
        return np.array([b.standard_deviation() for b in self.running_stats])
#        return np.sqrt(self.sum2/self.n - (self.mean*self.mean))

    @property
    def mean(self):
        return np.array([b.mean() for b in self.running_stats])
#        return self.sum / self.n

    def save(self, filename="TTS_dat.dat", header="None\nNone"):
        x = np.linspace(0, self.max_time, num=len(self.sum))
        y = self.mean
        arr = np.concatenate((np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)), axis=1)
        with open(filename, 'w') as F:
            F.write(header + "\n")
            _ = np.savetxt(fname=F, X=arr)

    def plot(self):
        if self._first_plot:
            plt.ion()
            self._plot_data, = self._ax.plot(np.linspace(0, self.max_time, num=len(self.sum)), np.log(0.01) / np.log(1.-self.mean), '.--', label="Probability of at least one success")
            self._plot_data_linear, = self._ax.plot(np.linspace(0, self.max_time, num=len(self.sum)), 10.*self.mean, '.--', color='gray', alpha=0.3, label="Probability of at least one success")
#            self._plot_data_sd_p, = self._ax.plot(np.linspace(0, self.max_time, num=len(self.sum)), self.mean+self.sd, '-', label="Probability of at least one success")
#            self._plot_data_sd_m, = self._ax.plot(np.linspace(0, self.max_time, num=len(self.sum)), self.mean-self.sd, '-', label="Probability of at least one success")
            self._ax.set_ylim([0, 10])
            self._ax.axhline(y=1, color='k', linestyle='--')
            self._ax.set_ylabel("Probability of at least one success")
            self._ax.set_xlabel("Time")
            self._fig.canvas.draw()
            plt.figure(self._fig.number)
            #plt.show()
            self._first_plot = False

        elif not self._first_plot:
            self._plot_data.set_ydata(np.log(0.01) / np.log(1-self.mean))
            self._plot_data_linear.set_ydata(10*self.mean)
 #           self._plot_data_sd_p.set_ydata(self.mean+self.sd)
 #           self._plot_data_sd_m.set_ydata(self.mean-self.sd)
            self._ax.relim()
            self._ax.autoscale_view(True, True, True)
            self._fig.canvas.draw()
            plt.pause(0.00001)

    def close(self):
        plt.close(fig=self._fig.number)


if __name__=='__main__':
    TL = TTSLogger(max_time=60, time_bins=100)
    N = 1000 #num test points
    for t in np.linspace(0, 58.99, num=N) + np.random.rand(N):
        rnd = np.random.normal(loc=30.0, scale=60.0)
        result = np.zeros(100, dtype=np.float32)
        if rnd < t:
            result += 1
        TL.record_frame(result=result, goal=1.0, time=t)

        TL.plot()



import networkx as nx
#import matplotlib
import numpy as np

class GraphFromLatfile():


    def __init__(self, latfile, L=4):
        G = nx.Graph()
        with open(latfile, 'r') as F:
            data = F.readlines()
        N = int(data[0])
        for line in data[1:]:
            i,j,Hij = line.split()
            i = int(i)
            j = int(j)
            Hij = float(Hij)
            if i==j:
                G.add_node(i, bias=Hij, pos=(i//L+np.random.rand()*0.3,i%L+np.random.rand()*0.3))
            if i != j:
                G.add_edge(i, j, coupling=Hij)
        #G = nx.grid_2d_graph(8, 8) 
        pos=nx.get_node_attributes(G,'pos') 
        self.G = G
        self.pos = pos

    def energy(self, spins):
        """This actually evaluates the negative of the Ising energy; apparently
           optimization people don't up the negative in the Ising energy."""
        graph = self.G
        E = 0.0
        edges = nx.get_edge_attributes(graph, 'coupling')
        for (i,j), coupling in edges.items():
            E += spins[i]*spins[j]*coupling

        h = nx.get_node_attributes(graph, 'bias')
        for i, spin in spins.items():
            E += h[i]*spins[i]

        return E / len(spins)


    def draw(self, conf, edges=False, mode=None):
        import matplotlib
        raw_spins = [conf[key] for key in conf.keys()]
        
        
        edge_weights = [5*abs(self.G[u][v]['coupling']) for u,v in self.G.edges]
        edge_colors = [plt.rcParams['axes.prop_cycle'].by_key()['color'][3] if self.G[u][v]['coupling']<0 else plt.rcParams['axes.prop_cycle'].by_key()['color'][4] for u,v in self.G.edges]
        nx.draw(self.G, self.pos, edges=self.G.edges, width=edge_weights, node_size=200,
                node_color=[plt.rcParams['axes.prop_cycle'].by_key()['color'][2] if i==1 else 'white' for i in raw_spins],
                edgecolors='k',
                edge_color=edge_colors,#plt.rcParams['axes.prop_cycle'].by_key()['color'][3],
               )
        
        #nx.draw_networkx_edges(self.G, self.pos, alpha=1.0, width=weights, edge_color='red')

