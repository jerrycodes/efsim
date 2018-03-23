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

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_face.h>

#include "../source/Velocity.h"
#include "../source/TransportSolver.h"
#include "../source/HelpFunctions.h"

#include <iostream>
#include <cstdlib>

using namespace dealii;

/*
 * Test TransportSolver
 * Runs one time step of transport solver for different grid refinements.
 * For the last refinement, we check if error norm is equal to reference.
 */

int main()
{
  // Suppress some output
  deallog.depth_console(0);

  {
	  // Problem spesific parameters
	  const int dim = 2;
	  const ProblemType pt = ProblemType::SANGHYUN;
	  const int last_refinement = 5;
	  const double dt = 0.05;
	  const double norm_tol = 1e-10;

	  // Needed to change this since with RT velocity, flux is constant on face so that a face is either completely inflow or outflow
	  const double reference_l2_norm = 0.011815955496092234514;
	  double l2_norm;

	  for (int refinement=2; refinement<last_refinement+1; ++refinement) {
		  std::cout << std::endl << "------------------------------"
					<< std::endl << "1/h = " << pow(2,refinement)
					<< std::endl << "------------------------------"
					<< std::endl;

		  // Set up grid
		  Triangulation<dim> triangulation;
		  make_grid(triangulation, pt, refinement);

		  RockProperties<dim> rock(triangulation);
		  rock.initialize(pt);
		  ProblemFunctionsFlow<dim> flow_fun(pt);
		  ProblemFunctionsTransport<dim> funs(pt);

		  // Initialize transport object and run one step
		  FE_DGQ<dim> fe_conc(0);
		  DoFHandler<dim> dh_conc(triangulation);

		  flow_fun.set_time(dt);
		  VelocityData<dim> velocity(triangulation);
		  velocity.setup_system();
		  velocity.init_exact(flow_fun);

		  TransportSolver<dim> transport(triangulation, fe_conc, dh_conc, rock, funs, pt);
		  transport.set_runtime_parameters("solution_transport_test", 2, 1.0);
		  transport.setup_system();
		  transport.set_velocity(velocity);
		  transport.solve_time_step(dt);

		  // Get error norms
		  l2_norm = transport.get_l2_error_norm();
		  std::cout.precision(20);
		  std::cout << "L2 norm error: " << l2_norm << std::endl;

		  dh_conc.clear();
	  }

	  std::stringstream l2_error_message;
	  l2_error_message << "\nERROR: L2 norm of concentration error is not equal to reference "
			  	  	   << "solution within an absolute tolerance of " << norm_tol
					   << ".\nCheck input parameters and newest implementations!\n";
	  if (abs(l2_norm - reference_l2_norm) > norm_tol) {
		  std::cerr << l2_error_message.str();
		  return 1;
	  }
  }

  return 0;
}
