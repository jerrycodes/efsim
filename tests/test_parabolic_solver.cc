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

#include "../source/ParabolicPressureSolver.h"
#include "../source/HelpFunctions.h"

using namespace dealii;

/*
 * Test ParabolicPressureSolver
 * Runs elliptic diffusion solver for several refinement levels.
 * For the last refinement, we check if flux error norm is equal to reference.
 */
int main (int argc, char *argv[])
{
	deallog.depth_console (0);
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

	const ProblemType pt = ProblemType::ANALYTIC;

	const double norm_tol = 1e-10;
	const double reference_l2_norm = 0.0019316371905745935867;
	const double dt = 0.01;
	const double end_time = 0.11;
	const int refinement=3;
	const int dim = 2;

	// Create grid
	Triangulation<dim> triangulation;
	make_grid(triangulation, pt, refinement, false, 0.0);

	ProblemFunctionsFlow<dim> flow_fun(pt);
	RockProperties<dim> rock(triangulation);
	rock.initialize(pt);

	// Create FE and dofs
	const FE_Q<dim> fe_pressure(1);
	DoFHandler<dim> dh_pressure(triangulation);

	std::cout.precision(20);

	// Initialize solver and run it
	ParabolicPressureSolver<dim> pressure_solver(triangulation, fe_pressure, dh_pressure, rock, flow_fun);
	pressure_solver.set_parameters(10.0, "solution_pressure", 2, true);
	pressure_solver.run(dt, end_time);
	double l2_norm = pressure_solver.pressure_l2_error();

	// Compare with reference solution
    std::stringstream l2_error_message;
  	l2_error_message << "\nERROR: L2 norm of pressure error is not equal to reference "
		  	  	     << "solution within an absolute tolerance of " << norm_tol
				     << ".\nCheck input parameters and newest implementations!\n";
  	if (abs(l2_norm - reference_l2_norm) > norm_tol) {
	    std::cerr << l2_error_message.str();
	    return 1;
    }

	return 0;
}
