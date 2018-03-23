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

#include "../source/PostProcessGS.h"
#include "../source/PostProcessMM.h"
#include "../source/EllipticPressureSolver.h"

#include <iostream>
#include <cstdlib>

using namespace dealii;

/*
 * Test PostProcessMM
 */


int main(int argc, char *argv[])
{
	// Suppress some output
	deallog.depth_console(0);
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

	const int dim = 2;
	const double tol = 1e-10;
	
	std::cout << "Test 1: Reference configuration" << std::endl;
	{
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

		RockProperties<dim> rock(triangulation);
		rock.initialize(ProblemType::SIMPLE_ANALYTIC);
		
		// Construct residual
		Vector<double> residual(triangulation.n_active_cells());
		residual = 0.0;
		residual(0) = 2.0;
		residual(1) = -1.0;
		residual(3) = 0.5;
		residual(5) = -1.0;
		residual(6) = 1.0;

		// Reference flux
		Vector<double> reference_correction(22);
		reference_correction(0) = 0.2015850815135474261;
		reference_correction(1) = -0.46789951178992172931;
		reference_correction(2) = 0.0;
		reference_correction(3) = -0.34787085046854177017;
		reference_correction(4) = -0.082286057167594572892;
		reference_correction(5) = 0.20231471848897453203;
		reference_correction(6) = -0.20622287225080707107;
		reference_correction(7) = 0.0;
		reference_correction(8) = -0.12002866132137997301;
		reference_correction(9) = -0.0;
		reference_correction(10) = 0.14508084745199836818;
		reference_correction(11) = 0.14580930359184021694;
		reference_correction(12) = 0.0;
		reference_correction(13) = -0.13894592679448339312;
		reference_correction(14) = 0.17700936074608314419;
		reference_correction(15) = 0.0;
		reference_correction(16) = -0.049151441534928591581;
		reference_correction(17) = 0.0061349206575149776641;
		reference_correction(18) = 0.23560378885139501848;
		reference_correction(19) = -0.088420977825109547954;
		reference_correction(20) = 0.22616080228101173577;
		reference_correction(21) = -0.32402476667650459419;

		FE_FaceQ<dim> fe_face(0);
		DoFHandler<dim> dh_face(triangulation);

		PostProcessMM<dim> postprocess(triangulation, fe_face, dh_face, rock);
		postprocess.set_parameters(tol);
		postprocess.setup_system();
		Vector<double> correction = postprocess.apply(residual);

		bool equal = true;
		for (unsigned int i=0; i<correction.size(); ++i) {
			if (abs(correction(i) - reference_correction(i)) > tol)
				equal = false;
		}

		/*
		std::cout.precision(20);
		for (unsigned int i=0; i<correction.size(); ++i)
			std::cout << correction(i) << std::endl;
		*/

		std::stringstream error_message;
		error_message << "\nERROR: Postprosessed flux correction is not equal to reference "
						<< "solution within an absolute tolerance of " << tol
						<< ".\nCheck input parameters and newest implementations!\n";
		if (! equal) {
			std::cerr << error_message.str();
			return 1;
		}

		dh_face.clear();
	}
	
	std::cout << std::endl << "Test 2: Embedded fracture surface" << std::endl;
	{
		const double perm_frac = 10.0;
		const ProblemType pt = SIMPLE_FRAC;
		const bool local_refinement = true;
		const unsigned int init_refinement = 2;
		const bool weighted_postprocessing = true;
		
		Triangulation<dim> triangulation;
		make_grid(triangulation, pt, init_refinement, local_refinement, 0.0);
		
		RockProperties<dim> rock(triangulation);
		rock.initialize(pt);
		ProblemFunctionsFlow<dim> flow_fun(pt);
		
		FractureNetwork fracture_network(pt);
		fracture_network.init_fractures(triangulation);
		fracture_network.output_to_vtk("test_fracture");
		
		const FE_Q<dim> fe_pressure(1);
		DoFHandler<dim> dh_pressure(triangulation);
		FE_FaceQ<dim> fe_correction(0);
		DoFHandler<dim> dh_correction(triangulation);
		
	    EllipticPressureSolver<dim> pressure_solver(triangulation, fe_pressure, dh_pressure, rock, flow_fun);
		pressure_solver.set_parameters(10.0, "solution_pressure", 2, false);
		pressure_solver.set_fractures(fracture_network, perm_frac, 1.0);
		pressure_solver.run();
		
		VelocityData<dim> velocity(triangulation);
		velocity.setup_system();
		pressure_solver.calculate_velocity(velocity, weighted_postprocessing);
		pressure_solver.add_fracture_velocity(velocity, 1.0);
		
		Vector<double> residuals(triangulation.n_active_cells());
		velocity.calculate_residuals(residuals, flow_fun, fracture_network);
		
		PostProcessMM<dim> postprocessor(triangulation, fe_correction, dh_correction, rock);
		postprocessor.set_parameters(tol, false, true, weighted_postprocessing);
		postprocessor.set_fractures(perm_frac);
		postprocessor.setup_system();
		
		Vector<double> correction = postprocessor.apply(residuals);
		velocity.add_correction(correction);
		velocity.calculate_residuals(residuals, flow_fun, fracture_network);
		
		if (! velocity.is_locally_conservative())
			return 1;
	}

  return 0;
}
