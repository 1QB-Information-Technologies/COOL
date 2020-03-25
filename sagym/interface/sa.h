/******************************************************************************

Simulated Annealing

Copyright (C) 2017-2020 1QBit
Contact info: Pooya Ronagh <pooya@1qbit.com>

This code is a modification of the 2012-2013 code of Sergei Isakov (Google).
The rights and license is hence inherited from the original as GNU General
Public License (v3). Original license follows.

---------------------------------------------------------------------

Copyright (C) 2012-2013 by Sergei Isakov <isakov@itp.phys.ethz.ch>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*******************************************************************************/

void get_lattice(double* arr, int size);
void get_all_energies(double* arr, int size);
int get_num_reps();
int get_num_spins();
float get_average_energy();
float get_acceptance_ratio();

void set_current_beta(double beta);
int reset(double beta);
int run(unsigned int arg_nsweeps, double end_beta);
double get_current_beta();
float get_average_absolute_magnetization();
