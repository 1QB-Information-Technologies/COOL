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

#ifndef __RANDOM_NUMBER_GENERATOR_H__
#define __RANDOM_NUMBER_GENERATOR_H__

#include <cmath>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

using namespace std;

/**
 * Random number generator for SQA implementations.
 */
template <typename engine_type>
class random_number_generator {

public:

    void seed(typename engine_type::result_type value) {
        _generator.seed(value);
    }

    bool next_bool() {
        return _next_bool(_generator);
    }

    int next_spin() {
        return next_bool() ? 1 : -1;
    }

    bool next_bernoulli(double probability) {
        boost::random::bernoulli_distribution<double> bernoulli =
            boost::random::bernoulli_distribution<double>(probability);
        return bernoulli(_generator);
    }

    double next_uniform_real(double lower_limit, double upper_limit) {
        boost::random::uniform_real_distribution<double> uniform =
            boost::random::uniform_real_distribution<double>(lower_limit, upper_limit);
        return uniform(_generator);
    }

    size_t next_uniform_integer(int lower_limit, int upper_limit) {
        boost::random::uniform_int_distribution<size_t> uniform =
            boost::random::uniform_int_distribution<size_t>(lower_limit, upper_limit);
        return uniform(_generator);
    }

    double next_poisson_point_interval(double parameter) {
        return -1*log(1-next_uniform_real(0,1))/parameter;
    }

    vector<double> next_poisson_point_process(
        double parameter,
        double lower_limit,
        double upper_limit) {

        vector<double> result;
        double value = lower_limit;

        while ((value += next_poisson_point_interval(parameter)) <= upper_limit) {
            result.push_back(value);
        }

        return result;
    }

private:

    engine_type _generator;
    boost::random::bernoulli_distribution<double> _next_bool =
        boost::random::bernoulli_distribution<double>(0.5);
};

#endif
