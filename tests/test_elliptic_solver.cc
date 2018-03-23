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

#include "../source/EllipticPressureSolver.h"
#include "../source/HelpFunctions.h"
#include "../source/Velocity.h"


using namespace dealii;

/*
 * Test EllipticPressureSolver
 * Runs elliptic diffusion solver for several refinement levels.
 * For the last refinement, we check if pressure L2 error norm is equal to reference.
 */

int main (int argc, char *argv[])
{
	deallog.depth_console(0);
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

	{
		const ProblemType pt = ProblemType::SIMPLE_ANALYTIC;

		const double norm_tol = 1e-10;
		const double reference_l2_norm = 0.0015525801667412844273;
		double l2_norm;

		for (int refinement=2; refinement<6; ++refinement)
		{
			std::cout << std::endl << "------------------------------"
					<< std::endl << "1/h = " << pow(2,refinement)
					<< std::endl << "------------------------------"
					<< std::endl;
			const int dim = 2;
			Triangulation<dim> triangulation;
			make_grid(triangulation, pt, refinement, false, 0.0);

			RockProperties<dim> rock(triangulation);
			rock.initialize(pt);
			ProblemFunctionsFlow<dim> flow_fun(pt);

			const FE_Q<dim> fe_pressure(1);
			DoFHandler<dim> dh_pressure(triangulation);

			EllipticPressureSolver<dim> laplace_problem_2d(triangulation, fe_pressure, dh_pressure, rock, flow_fun);
			laplace_problem_2d.set_parameters(10.0, "solution_pressure", 2, true);
			laplace_problem_2d.run();

			VelocityData<dim> velocity(triangulation);
			velocity.setup_system();
			laplace_problem_2d.calculate_velocity(velocity, false);
			Vector<double> residuals(triangulation.n_active_cells());
			velocity.calculate_residuals(residuals, flow_fun.right_hand_side);
			velocity.write_to_vtk("velocity_test");

			//std::cout.precision(20);
			l2_norm = laplace_problem_2d.pressure_l2_error();
			std::cout << "L2 pressure error: " << l2_norm << std::endl;
		}

		std::stringstream l2_error_message;
		l2_error_message << "\nERROR: L2 norm of pressure error is not equal to reference "
						<< "solution within an absolute tolerance of " << norm_tol
						<< ".\nCheck input parameters and newest implementations!\n";
		if (abs(l2_norm - reference_l2_norm) > norm_tol) {
			std::cerr << l2_error_message.str();
			return 1;
		}
	}
	
	// Test with stabilization
	{
		const ProblemType pt = SIMPLE_FRAC;
		const unsigned int dim = 2;
		const double frac_perm = 1e2;
		
		Triangulation<dim> tria;
		make_grid(tria, pt, 2);
		
		RockProperties<dim> rock(tria);
		rock.initialize(pt);
		ProblemFunctionsFlow<dim> flow_fun(pt);
		
		const FE_Q<dim> fe_pressure(1);
		DoFHandler<dim> dh_pressure(tria);
		
		FractureNetwork fractures;
		fractures.add(EmbeddedSurface<dim>(Point<dim>(1.0/3.0,1.0), Point<dim>(1.0,1.0/3.0)));
		fractures.init_fractures(tria);
		fractures.output_to_vtk("fracture_stab_test");

		EllipticPressureSolver<dim> pressure_solver(tria, fe_pressure, dh_pressure, rock, flow_fun);
		pressure_solver.set_parameters(10.0, "solution_pressure_stab_test", 2, false);
		pressure_solver.set_fractures(fractures, frac_perm, 1.0);
		pressure_solver.set_stabilization(1.0);
		
		pressure_solver.run();
		
	}

	return 0;
}
