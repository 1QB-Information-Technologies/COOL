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

def ising_get_neighbours(i, L):
    """Given index i, and lattice length L (i.e. L x L ising model),
       calculate the up, down, left, and right nearest neighbours
    """
    row = math.floor(i/L)
    start = row*L
    l=(i-L*row-1)%L+L*row
    r=(i-L*row+1)%L+L*row
    u= (i-L)%(L*L)
    d=(i+L)%(L*L)
    return u,l,d,r

def one():
    """Return one as a float"""
    return 1.0

def make_ferro_ising(latfile_path, L=4, J=one):
    with open(latfile_path, 'w') as F:

        F.write(f"{L**2}\n")
        
        for i in range(L**2):
            """For each site, write explicitly the bias."""
            F.write(f"{i} {i} 0.0 \n")

        for i in range(L**2):
            """For each site, calculate the neighbours"""
            ns = ising_get_neighbours(i, L)
            for neighbour in [ns[0], ns[1]]:
                """For each neighbour, write an entry in the latfile. Note -J is written, not J"""

                F.write(f"{i} {neighbour} {-J()} \n")


def normal():
    return np.random.normal(loc=0, scale=1.0)

def truncated_normal():
    x=-1000000
    while x < -1 or x > 1:
        x = np.random.normal(loc=0, scale=0.5)
    return x

def make_nn_ising(latfile, J, L=4, seed=None):
    if seed is not None:
        np.random.seed(seed)
    #Make the latfile:
    make_ferro_ising(latfile, L=L, J=J)



def make_rndj_nn_sg(latfile, L, seed=None):
    make_nn_ising(latfile=latfile, J=truncated_normal, L=L, seed=seed)
    
def make_rndj_notrunc_nn_sg(latfile, L, seed=None):
    make_nn_ising(latfile=latfile, J=normal, L=L, seed=seed)
