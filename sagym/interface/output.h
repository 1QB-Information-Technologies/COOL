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

#ifndef __OUTPUT_H__
#define __OUTPUT_H__

#include <map>
#include <unordered_map>
#include <string>
#include <vector>

std::string positive_spin_indices(const std::string& spin_vector,
  const std::vector<size_t>& all_indices) {

  std::stringstream result;
  result << "[";

  for (size_t index=0;  index < spin_vector.length(); index++) {
    if (spin_vector[index] == '+') {
      result << all_indices[index] << ", ";
    }
  }

  result << "\b\b]";
  return result.str();

}

void write_results(
  const std::vector<std::pair<double, std::string> >& en,
  const std::string& latfile,
  const std::vector<size_t>& all_indices,
  unsigned nreps) {

  std::map<std::string, size_t> spin_counts;
  std::multimap<size_t, std::string> count_spins;
  std::unordered_map<std::string, double> spin_energies;

  for (const auto& sample_outcome : en) {

    double energy = sample_outcome.first;
    std::string spin_vector = sample_outcome.second;
    spin_counts[spin_vector]++;
    spin_energies[spin_vector] = energy;

  }

  for (const auto& spin_count : spin_counts) {

    size_t count = spin_count.second;
    std::string spin = spin_count.first;
    count_spins.insert(std::pair<size_t, std::string>(count, spin));

  }

  for (auto it = count_spins.rbegin(); it != count_spins.rend(); ++it) {

    auto& result = *it;
    std::string spin_vector = result.second;
    size_t count = result.first;
    double energy = spin_energies[spin_vector];

    std::cout << energy
      << "\t" << spin_vector
      << "\t" << (double)count/en.size()
      << std::endl;
  }

}

#endif