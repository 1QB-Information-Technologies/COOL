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

#ifndef __SCHED_H__
#define __SCHED_H__

#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

/**
 * Generates a new schedule.
 *
 * @param sched_kind May be "lin" for linear schedule or "exp" for exponential. If
 * sched_kind has any other value, then it is treated as a file name from where to read
 * in the schedule.
 * @param nsweeps    The number of time steps in the schedule
 * @beta0            The value of the inverse temperature at the start of the schedule.
 * @beta1            The value of the inverse temperature at the end of the schedule.
 *
 * @return           A new schedule stored as a vector of sched_entries.
 */
inline std::vector<double> get_sched (const std::string sched_kind,
  unsigned nsweeps, double beta0, double beta1) {

  std::vector<double> sched;

  if (sched_kind == "lin") {

    sched.resize(nsweeps);
    double bscale = nsweeps > 1 ? (beta1 - beta0) / (nsweeps - 1) : 0.0;

    for (size_t i = 0; i < nsweeps; ++i) {
      sched[i] = beta0 + bscale * i;
    }

  } else if (sched_kind == "exp") {

    sched.resize(nsweeps);
    sched[0] = beta0;
    double db = pow(beta1 / beta0, 1.0 / (nsweeps - 1));

    for (size_t i = 1; i < nsweeps; ++i) {
      sched[i] = sched[i - 1] * db;
    }

  } else {

    std::ifstream fin;
    fin.open(sched_kind.c_str(), std::ios_base::in);

    if (!fin) {
      throw std::runtime_error("cannot open file " + sched_kind);
    }

    sched.reserve(10000);

    double beta;
    while (fin >> beta) {
      sched.push_back(beta);
    }

    fin.close();

  }

  return sched;
}

#endif

