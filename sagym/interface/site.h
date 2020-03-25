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

#ifndef __SITE_H__
#define __SITE_H__


/**
 * Each site stores information about neighboring sites.
 */
struct neighbor_type {

  size_t id;

  double coupler;

  neighbor_type(size_t id, const double coupler) :
    id(id), coupler(coupler) {
  }

};


/**
 * Stores data associated with a single spin.
 */
struct site_type{

  int spin;

  /**
  * Coefficients of the linear terms
  */
  double local_field_bias;

  /**
  * Local field
  */
  double de;

  /**
  * Count of neighboring sites.
  */
  size_t nneighbs;

  /**
  * List of neighboring sites
  */
  std::vector<neighbor_type> neighbors;

};

struct Link {

  size_t s0;

  size_t s1;

  double val;

  Link(size_t s0, size_t s1, double val) :
      s0(s0), s1(s1), val(val) {
  }

  bool operator<(const Link& rhs) const {
    return fabs(val) < fabs(rhs.val);
  }

};

#endif