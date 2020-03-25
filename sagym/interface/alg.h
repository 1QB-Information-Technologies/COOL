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

#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#include <cmath>
#include <vector>
#include <string>
#include <boost/random/shuffle_order.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ctime>

#include "site.h"
#include "lattice.h"
#include "random_number_generator.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>




class Algorithm {

public:

  Algorithm() {}

  double beta;
  int accepts=0;
  int totals=0;


  Algorithm(const Lattice& lattice)
  {
     lattice.init_sites(sites);
  }


  void set_negative() {
    for(auto& site : sites)
      site.spin = -1;
    for(auto& site : sites) {
      double tmp = site.local_field_bias;
      for(size_t k = 0; k < site.nneighbs; ++k) {
        tmp -= site.neighbors[k].coupler;
      }
      site.de = -tmp * site.spin;
    }
  }


  


  void reset_sites(size_t rep) {

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    _random.seed((rep+1)*(time_t)ts.tv_nsec);

//_random.seed((rep+1) * time( NULL ) );

    ifstream File;
    File.open("latinit");
    int tmp;
    int i;
    for(auto& site : sites)
    {
       File >> tmp; //tmp = File.getline(); 
       if (tmp==1) {
//           cout << "Read spin 1 from file " << endl; 
           site.spin = 1;
       } else if (tmp==-1) {
//           cout << "Read spin -1 from file " << endl; 
           site.spin = -1;
       } else {
        site.spin = _random.next_bool() ? 1 : -1;
       }
    }
    File.close();
  

    for(auto& site : sites) {

      double tmp = site.local_field_bias;
      for(size_t k = 0; k < site.nneighbs; ++k) {
        tmp += site.neighbors[k].coupler * sites[site.neighbors[k].id].spin;
      }
      site.de = -tmp * site.spin;

    }

  }

  void flip_spin(site_type& site) {

    site.spin = -site.spin;
    site.de = -site.de;

    for (size_t k = 0; k<site.nneighbs; ++k) {
      site_type& neighbor = sites[site.neighbors[k].id];
      neighbor.de -= 2 * neighbor.spin * site.neighbors[k].coupler * site.spin;
    }

  }


  void reset_acceptance() {
    accepts=0;
    totals=0;
  }


  void do_sweep(size_t sweep) {

    for (size_t i = 0; i < sites.size(); ++i) {
      size_t next_index= _random.next_uniform_integer(0, sites.size() - 1);
      //cout << "beta=" << *beta << endl;
      if (sites[next_index].de <
        -log(_random.next_uniform_real(0,1)) / (beta * 2)) {
        flip_spin(sites[next_index]);
	accepts++;
      } 
      totals++;
    }

  }

  string get_configuration() {
    string configuration;
    for (size_t i=0; i<sites.size(); ++i) {
      if (sites[i].spin>0) {
        configuration+="+";
      } else {
        configuration+="-";
      }
    }
  return configuration;
  }


  int get_site(int n) {
    return sites[n].spin;
  }

  int num_sites() {
    return sites.size();
  }


  float get_energy() const {

    double energy = 0;
    for(const auto& site : sites) {

      double tmp = site.local_field_bias;

      for(size_t k = 0; k < site.nneighbs; ++k) {
        tmp += sites[site.neighbors[k].id].spin * site.neighbors[k].coupler / 2;
      }
      energy += tmp * site.spin;
    }

    return energy;
  }

  void get_energies(vector<pair<double, string> >& en, size_t index) const {

    string configuration;
    double energy = 0;
    for(const auto& site : sites) {

      if (site.spin>=0) {
        configuration+="+";
      } else {
        configuration+="-";
      }
      double tmp = site.local_field_bias;

      for(size_t k = 0; k < site.nneighbs; ++k) {
        tmp += sites[site.neighbors[k].id].spin * site.neighbors[k].coupler / 2;
      }

      energy += tmp * site.spin;
    }

    en[index].first = energy;
    en[index].second= configuration;

  }

private:

  std::vector<site_type> sites;

  std::vector<double> beta_schedule;

  //random_number_generator<boost::random::knuth_b> _random;
  random_number_generator<boost::random::mt19937> _random;

};

#endif
