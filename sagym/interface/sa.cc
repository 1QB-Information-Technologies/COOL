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

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <tclap/CmdLine.h>
#include <tclap/StdOutput.h>
#include <thread>
#include "sched.h"
#include "output.h"
#include "sa.h"
#include "lattice.h"
#include "alg.h"

#ifdef _OPENMP
#include "omp.h"
#endif

using namespace std::chrono;
using namespace TCLAP;


string arg_latfile="./latfile";
unsigned int arg_nreps=64;


void incr_current_beta(double incr, size_t rep) ;


/*##############################################
################################################
################################################
################################################
################################################*/


#ifdef _OPENMP
    unsigned int arg_n_threads=omp_get_max_threads();
#endif


#ifndef _OPENMP
    const unsigned n_threads=1;
#else
    const unsigned n_threads= arg_n_threads;
#endif

// read lattice
Lattice lattice(arg_latfile);

vector<Algorithm> make_algorithm_objects(Lattice lattice__) {
    vector<Algorithm> alg(arg_nreps);

    for (unsigned int rep=0; rep<arg_nreps; rep++) {
      alg[rep] = Algorithm(lattice__);
    }
    return alg;
}

vector<Algorithm> alg = make_algorithm_objects(lattice);

/*##############################################
################################################
################################################
################################################
################################################*/

int reset(double beta) {
    // read lattice
    Lattice lattice(arg_latfile);
    alg = make_algorithm_objects(lattice);
    //double beta = 0.00001;
    bool arg_neg_init = false; //("z", "neg_init", "Negative initialization", false);

      for (size_t rep=0; rep < arg_nreps; rep++) {
        alg[rep].beta = beta;
        if (arg_neg_init) {
          alg[rep].set_negative();
        } else {
          alg[rep].reset_sites(rep);
        }
      }
    return 0;
}


int run(unsigned int arg_nsweeps, double dbeta) {

//    std::vector<std::pair<double, std::string> > en(arg_nreps);

#ifndef _OPENMP
    const unsigned n= 0;
#else
    #pragma omp parallel num_threads(n_threads)
    {
      const unsigned n = omp_get_thread_num();
#endif


      size_t r0 = arg_nreps * n / n_threads;
      size_t r1 = arg_nreps * (n + 1) / n_threads;


      //check to make sure the temperature update is legal.
      //set dbeta to 0 if illegal so T is not updated during
      //sweeps
      if (alg[0].beta + dbeta < 0.000001) {
	      dbeta = 0.0;
      }


      // main loop
      for (size_t rep = r0; rep < r1; rep++) {
        for (size_t sweep = 0; sweep < arg_nsweeps; ++sweep) {
          incr_current_beta(dbeta / arg_nsweeps, rep);
          alg[rep].do_sweep(sweep);
        }
//        alg[rep].get_energies(en, rep);
      }

#ifdef _OPENMP
    }
#endif

  return 0;

}


float get_acceptance_ratio() {
    int accepted_flips=0;
    int total_flips=0;
    for (size_t rep=0; rep<arg_nreps; rep++) {
	    accepted_flips += alg[rep].accepts;
	    total_flips += alg[rep].totals;
	    alg[rep].reset_acceptance();
    }
    
    return float(accepted_flips) / float(total_flips);
    
}


float get_average_energy() {
    float summ = 0;
    for (size_t rep=0; rep<arg_nreps; rep++) {
        summ += alg[rep].get_energy();
    }
    return summ / arg_nreps / alg[0].num_sites();
}


float get_average_absolute_magnetization() {
  float configSum=0;
  float this_config_sum;
  for (unsigned int rep=0; rep<arg_nreps; rep++) {
    //cout << rep << alg[rep].get_configuration();
    float spinSum=0;
    for (int i=0; i<alg[rep].num_sites(); i++){
      spinSum += alg[rep].get_site(i);
    }
    this_config_sum = abs(spinSum);
    configSum += this_config_sum / alg[rep].num_sites();
  }
  return configSum / alg.size();

}

int get_num_reps() {
  return arg_nreps;
}

int get_num_spins(){
  return alg[0].num_sites();
}


void get_all_energies(double* arr, int size) {
   for (int rep=0; rep<arg_nreps; rep++) {
       arr[rep] = alg[rep].get_energy();
   }


}

void print_lattice() {
   for (int rep=0; rep < arg_nreps; rep++) {
    for (int spin=0; spin<alg[rep].num_sites(); spin++) {
        cout << alg[rep].get_site(spin) << " ";
      }
      cout << endl; 
    }
  }
 
  

void get_lattice(double* arr, int size) {
  int i=0;
  for (int rep=0; rep<arg_nreps; rep++) {
    for (int spin=0; spin<alg[rep].num_sites(); spin++) {
      if (alg[rep].get_site(spin) == 1 ) {
        arr[i] = 1;
      } else {
        arr[i] = -1;
      }
      //arr[i] = alg[rep].get_site(spin);
     // if (spin==0) {arr[i] = 1;}
      i++;
    }
  }
}


void set_current_beta(double beta) {
  for (size_t rep=0; rep<arg_nreps; rep++) {
    alg[rep].beta = beta ;
  }
}

double clip_zero(double T) {
  if (T<0) {
    return 0;
  } else {
    return T;
  }
}


void incr_current_beta(double incr, size_t rep) {
  if (alg[rep].beta + incr > 0.000001) {
    alg[rep].beta += incr;
  }
}

double get_current_beta() {
  double _beta = 0;
  for (size_t rep=0; rep<arg_nreps; rep++) {
    _beta += alg[rep].beta;
  }
  _beta = _beta / arg_nreps;
  return _beta;
}
