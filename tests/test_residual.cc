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

#include "../source/HelpFunctions.h"
#include "../source/EllipticPressureSolver.h"
#include "../source/Velocity.h"

#include <iostream>
#include <cstdlib>

using namespace dealii;

/*
 * Test Residual
 */

int main(int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

	const int dim = 2;
	const ProblemType pt = ProblemType::SIMPLE_ANALYTIC;
	const int dirichlet_penalty = 10.0;
	const double norm_tol = 1e-10;
	const bool weak_bcs = true;

	// Suppress some output
	deallog.depth_console(0);

	// Set up grid
	Triangulation<dim> triangulation;
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);
	std::ifstream grid_input("grid_four_cells_distorted.ucd");
	if (grid_input.is_open())
		grid_in.read_ucd(grid_input);
	else {
		std::cout << "Error opening ucd file\n";
		exit(1);
	}
	grid_input.close();

	triangulation.begin_active()->set_refine_flag();
	triangulation.execute_coarsening_and_refinement();

	ProblemFunctionsFlow<dim> flow_fun(pt);
	RockProperties<dim> rock(triangulation);
	rock.initialize(pt);

	// Solve fore pressure
	const FE_Q<dim> fe_pressure(1);
	DoFHandler<dim> dh_pressure(triangulation);
    EllipticPressureSolver<dim> laplace_problem_2d(triangulation, fe_pressure, dh_pressure, rock, flow_fun);
    laplace_problem_2d.set_parameters(dirichlet_penalty, "solution_pressure", 2, weak_bcs);
    laplace_problem_2d.run ();
    Vector<double> pressure = laplace_problem_2d.get_pressure_solution();

	// Calculate velocity
	VelocityData<dim> velocity(triangulation);
	velocity.setup_system();
	laplace_problem_2d.calculate_velocity(velocity, false);

    // Put up some correction vector
	FE_FaceQ<dim> fe_correction(0);
	DoFHandler<dim> dh_correction(triangulation);
	dh_correction.distribute_dofs(fe_correction);
	Vector<double> correction(dh_correction.n_dofs());
	for (unsigned int i=0; i<dh_correction.n_dofs(); ++i)
		correction(i) = sin(i);
	ConstraintMatrix constraints;
	setup_flux_constraints_subfaces(dh_correction, constraints, false);
	constraints.close();
	constraints.distribute(correction);

	// Calculate residual
	Vector<double> residuals(triangulation.n_active_cells());
	double residual_l2norm_no_corr   = velocity.calculate_residuals(residuals, flow_fun.right_hand_side);
	
	triangulation.clear_user_flags();
	velocity.add_correction(correction);
	triangulation.clear_user_flags();
	double residual_l2norm_with_corr = velocity.calculate_residuals(residuals, flow_fun.right_hand_side);

	double reference_l2norm_with_corr = 5.5684023193246741101;
	double reference_l2norm_no_corr   = 5.7748992774383616222;

	std::cout.precision(20);
	std::cout << "Residual L2 norm (with correction) : " << residual_l2norm_with_corr << std::endl;
	std::cout << "Residual L2 norm (no correction)   : " << residual_l2norm_no_corr << std::endl;

	std::stringstream l2_error_message;
	l2_error_message << "\nERROR: L2 norm of residuals is not equal to reference "
			         << "solution within an absolute tolerance of " << norm_tol
					 << ".\nCheck input parameters and newest implementations!\n";
	if ( (abs(residual_l2norm_with_corr - reference_l2norm_with_corr) > norm_tol) ||
		 (abs(residual_l2norm_no_corr   - reference_l2norm_no_corr)   > norm_tol) )
	{
		std::cerr << l2_error_message.str();
		return 1;
	}

	return 0;
}
