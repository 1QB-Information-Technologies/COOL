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

from sagym.interface import sa_interface as sa
import numpy as np

class SA(object):
    """This is a very thin wrapper class over the SA C++ interface,
       yltsom to document the interface and do thin error checking.
    """
    def reset(self, beta=None):
        """
        Reinitialize the SA lattice(s).  Note that the latfile Hamiltonian
          definition is read upon import, not upon reset, so the Hamiltonian
          will not be refreshed if it is changed on disk.

        Args:
            beta (float): Reset the (reciprocal) temperature to this value.
        """
        sa.reset(beta)

    def set_current_beta(self, beta=None):
        """
        Discontinuously set temperature to a specific floating point
            value (no schedule).

        Args:
            beta (float): the desired reciprocal temperature, i.e 1/(k_B*T),
                          with k_B=1
        """
        sa.set_current_beta(beta)


    def run(self, N_sweeps, dbeta=0.0):
        """
        Run some annealing sweeps.
        Args:
            N_sweeps (int): The number of annealing sweeps to perform.
            dbeta (float): The amount that beta should be changed over
            the course of N_sweeps sweeps.  beta will be changed by
            dbeta/N_sweeps before *every* sweep.
        """
        sa.run(N_sweeps, dbeta)


    def get_num_reps(self):
        """
        Return the number of reps in the current SA simulation

        Returns:
            reps (int): The number of reps
        """
        return sa.get_num_reps()

    def get_num_spins(self):
        """
        Return the number of spins in the current SA simulation lattices

        Returns:
            spins (int): The number of spins (i.e. lattice size)
        """
        return sa.get_num_spins()

    def set_lattice(self, lattice):
        """
        Sets the spins of the lattice to the values given in the array lattice

        Args:
            lattice (2d array, float): Array of size [reps, spins]

        """
        reps = self.get_num_reps()
        spins = self.get_num_spins()
        assert lattice.shape == (reps,spins), \
            "The provided spin values are not of shape [reps, spins]"

        sa.set_lattice(lattice.flatten(order='C')) #, lattice.size)
#        sa.set_lattice(lattice, reps*spins)


    def get_lattice(self):
        """
        Return the current spin configuration of the lattices (for all
        reps.

            Returns:
                S (2d array, float): Array of size [reps, spins].
        """
        reps = self.get_num_reps()
        spins = self.get_num_spins()
        lattices = sa.get_lattice(reps*spins)
        lattices = np.reshape(lattices, (reps, spins))
        return lattices

    def get_average_energy(self):
        """
        Return the average energy over the ensemble of spin lattices

            Returns:
                E (float): The average energy over the ensemble of lattices
        """

        return sa.get_average_energy()


    def get_all_energies(self):
        """
        Return the individual energies for each rep

            Returns:
                E (1d array, float): Array of energies of size [reps]

        """
        energies = sa.get_all_energies(self.get_num_reps())
        return energies

    def get_acceptance_ratio(self):
        """
        Return the acceptance ratio of the last call to run(),
        that is, (number of accepted spin flips) / (total flips proposed)

            Returns:
                r (float): acceptance ratio
        """
        ratio = sa.get_acceptance_ratio()
        return ratio



    def get_average_absolute_magnetization(self):
        """
        Return the current average absolute magnetization over
        the entire ensemble of lattices.
        <M> = 1/Nrep * 1/L * sum(abs(sum(sigma))
        Note this is an instantaneous measurement and no
        time averaging is done.

            Returns:
                M (float): The average absolute magnetization, as defined
                    above.
        """

        return sa.get_average_absolute_magnetization()

    def get_stddev_average_absolute_magnetization(self):
        """
        Return the current average absolute magnetization over
        the entire ensemble of lattices.
        <M> = 1/Nrep * 1/L * sum(abs(sum(sigma))
        Note this is an instantaneous measurement and no
        time averaging is done.

            Returns:
                M (float): The average absolute magnetization, as defined
                    above.
        """

        return sa.get_stddev_mean_absolute_magnetization()


    def get_current_beta(self):
        """
        Return the current value of beta, computed as the average over
        replicas.

        Returns:
            beta (float): the current value of beta

        """
        return sa.get_current_beta()
