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

#ifndef __LATTICE_H__
#define __LATTICE_H__

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#include "site.h"

/**
 * Graph representation of a quadratic unconstrained
 * binary optimization problem instance.
 * The problem, in general, is to optimize an objective function
 * \f[ E = \sum_{i<j} J_{i,j} s_i s_j + \sum_{i} h_i s_i \f]
 * where
 * \f$ s_i \in \{-1, +1\} \f$ are the spins,
 * \f$ J_{i,j} \in (-\infty,  +\infty) \f$ are the coupling weights
 * (linear weights),
 * \f$ J_{i,j} = J_{j,i} \f$ for all i,j;
 * and \f$ h_i \f$ are the linear weights.
 */

class Lattice {

public:

Lattice(const std::string& lattice_file) {

  std::ifstream fin;
  fin.open(lattice_file.c_str(), std::ios_base::in);

  if (!fin) {
    throw std::runtime_error("cannot open file "
      + lattice_file + " to read lattice");
  }

  maxs = 0;
  links.reserve(32768);

  std::string input_line;

  /*
  * Read the first line which contains the number of spins
  */
  fin >> input_line;

  /*
  * Read the remaining lines which have format:
  * s0, s1, cval
  */
  size_t s0, s1;

  double cval;

  while (fin >> s0 >> s1 >> cval) {

    links.push_back(Link(size_t(s0), size_t(s1), cval));

    maxs = s0 > maxs ? s0 : maxs;
    maxs = s1 > maxs ? s1 : maxs;

  }

  fin.close();

  nsites = 0;
  std::vector<size_t> phys_sites(maxs + 1, size_t(-1));
  _index_positions.clear();

  for (size_t i = 0; i < links.size(); ++i) {

    Link& link = links[i];

    if (phys_sites[link.s0] == size_t(-1)) {
      _index_positions.push_back(link.s0);
      link.s0 = phys_sites[link.s0] = nsites++;
    } else {
      link.s0 = phys_sites[link.s0];
    }

    if (phys_sites[link.s1] == size_t(-1)) {
      _index_positions.push_back(link.s1);
      link.s1 = phys_sites[link.s1] = nsites++;
    } else {
      link.s1 = phys_sites[link.s1];
    }

  }

  // need this for higher ranges
  sort(links.begin(), links.end());

}

void init_sites(std::vector<site_type>& sites) const {

  sites.reserve(nsites);
  sites.resize(nsites);

  for (size_t i = 0; i < links.size(); ++i) {
    const Link& link = links[i];

    if (link.s0 == link.s1) {

      sites[link.s0].local_field_bias = link.val;

    } else {

      sites[link.s0].neighbors.push_back(
        neighbor_type(link.s1, link.val));
      ++sites[link.s0].nneighbs;

      sites[link.s1].neighbors.push_back(
      neighbor_type(link.s0, link.val));
      ++sites[link.s1].nneighbs;

    }

  }

}

std::string print_index_positions() const {

  std::stringstream stream;
  stream << "[";

  for (const auto& index : _index_positions) {
    stream << index << ", ";
  }

  stream << "\b\b]";
  return stream.str();

}

const std::vector<size_t>& index_positions() const {
  return _index_positions;
}

private:

  size_t nsites;

  std::vector<Link> links;

  /**
  * Highest ID of any spin
  */
  size_t maxs;

  /**
  * Vector that shows how indices from the lattice file map to
  * positional indices actually used by the optimization algorithm
  * to represent spin vectors.
  * This is necessary because of faulty qubits.
  */
  std::vector<size_t> _index_positions;

};

#endif
