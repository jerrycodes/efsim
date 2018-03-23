/*
    Copyright (C) 2018 Lars Hov Odsæter

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


using namespace dealii;

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_cg.h>

#ifdef DEAL_II_WITH_TRILINOS
#define USE_TRILINOS
#endif

using namespace dealii;

namespace LA {

#ifdef USE_TRILINOS
	typedef TrilinosWrappers::SparseMatrix Matrix;
	typedef TrilinosWrappers::SolverCG CG;
	typedef TrilinosWrappers::PreconditionAMG AMG;
	typedef TrilinosWrappers::MPI::Vector Vec;
#else
	typedef SparseMatrix<double> Matrix;
	typedef SolverCG<> CG;
	typedef PreconditionSSOR<> SSOR;
	typedef Vector<double> Vec;
#endif

}