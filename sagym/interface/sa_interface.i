/*-----------------------------------------------------------------------------

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

-----------------------------------------------------------------------------*/



%module sa_interface
%{
    #define SWIG_FILE_WITH_INIT
    #include "sa.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* ARGOUT_ARRAY1, int DIM1) { (double* arr, int size)};
%apply (double* IN_ARRAY1, int DIM1) { (double* arr_in, int size)};

%include "sa.h"
