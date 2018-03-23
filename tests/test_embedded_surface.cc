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

#include "../source/EmbeddedSurface.h"
#include "../source/EllipticPressureSolver.h"
#include "../source/PostProcessMM.h"
#include "../source/TransportSolver.h"
#include "../source/ParameterHandlerFunctions.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/convergence_table.h>


using namespace dealii;


/*
 * Test EmbeddedSurface
 */


int main (int argc, char *argv[])
{
	deallog.depth_console(0);
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
	
	const int dim = 2;
	ProblemType pt = ProblemType::ONED;
	
	Triangulation<dim> tria;
	GridGenerator::hyper_cube(tria, 0, 1);
	tria.refine_global(1);
	
	typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active();
	cell->set_refine_flag();
	++cell; ++cell; ++cell;
	cell->set_refine_flag();
	tria.execute_coarsening_and_refinement();
	
	// Complex fracture
	{
		const double h = 0.25;
		std::vector<Point<dim>> vertices(13);
		vertices[0]  = Point<dim>(0.0,       h/3.0);
		vertices[1]  = Point<dim>(h/2.0,     h/3.0);
		vertices[2]  = Point<dim>(2.0*h/3.0, h/2.0);
		vertices[3]  = Point<dim>(h,         4.0*h/3.0);
		vertices[4]  = Point<dim>(3.0*h/2.0, 3.0*h/2.0);
		vertices[5]  = Point<dim>(5.0*h/3.0, 2.0*h/3.0);
		vertices[6]  = Point<dim>(7.0*h/3.0, 4.0*h/3.0);
		vertices[7]  = Point<dim>(5.0*h/2.0, 7.0*h/3.0);
		vertices[8]  = Point<dim>(8.0*h/3.0, 8.0*h/3.0);
		vertices[9]  = Point<dim>(10.0*h/3.0,7.0*h/2.0);
		vertices[10] = Point<dim>(7.0*h/2.0, 3.0*h);
		vertices[11] = Point<dim>(3.0*h,     2.0*h);
		vertices[12] = Point<dim>(4.0*h,     4.0*h/3.0);
		
		EmbeddedSurface<dim> fracture(vertices);
		fracture.initialize(tria);
		fracture.print_to_screen(tria);
		fracture.output_to_vtk("test_polygonal_chain", tria);
	}
	
	// Simple fracture coupled to flow
	{
		const double perm_fracture = 100.0;
		
		EmbeddedSurface<dim> fracture(Point<dim>(0.0,0.3), Point<dim>(1.0,0.3), 1);
		fracture.initialize(tria);
		fracture.output_to_vtk("test_straight_fracture");
		FractureNetwork fracture_network;
		fracture_network.add(fracture);
		
		RockProperties<dim> rock(tria);
		rock.initialize(pt);
		ProblemFunctionsFlow<dim> flow_fun(pt);
		
		const FE_Q<dim> fe_pressure(1);
		DoFHandler<dim> dh_pressure(tria);
		dh_pressure.distribute_dofs(fe_pressure);
		
		EllipticPressureSolver<dim> pressure_solver(tria, fe_pressure, dh_pressure, rock, flow_fun);
		pressure_solver.set_parameters(10.0, "solution_pressure_fracture", 2, false);
		pressure_solver.set_fractures(fracture_network, perm_fracture, 1.0);
		pressure_solver.run();
	}
	
	// Run analytical example from Burman2017asf
	{
		pt = FRACTURE_ANALYTIC;
		unsigned int refinement_level = 2;
		unsigned int n_qpoints = 4;
		const unsigned int max_refinement_level = 5;
		const double perm_fracture = 1.0;
		tria.clear();
		GridGenerator::hyper_cube(tria, 1, exp(5.0/4.0));
		tria.refine_global(refinement_level);
		
		VelocityData<dim> velocity(tria);
		
		FractureNetwork fracture_network(pt);
		
		const FE_Q<dim> fe_pressure(1);
		DoFHandler<dim> dh_pressure(tria);
		FE_FaceQ<dim> fe_correction(0);
		DoFHandler<dim> dh_correction(tria);
		
		RockProperties<dim> rock(tria);
		rock.initialize(pt);
		ProblemFunctionsFlow<dim> flow_fun(pt);
		
		EllipticPressureSolver<dim> pressure_solver(tria, fe_pressure, dh_pressure, rock, flow_fun);
		pressure_solver.set_parameters(10.0, "solution_pressure_burman_frac", n_qpoints, false);
		pressure_solver.set_fractures(fracture_network, perm_fracture, 1.0);
		
		PostProcessMM<dim> postprocessor(tria, fe_correction, dh_correction, rock);
		postprocessor.set_parameters(1e-12);
		
		Vector<double> residuals;
		
		ConvergenceTable conv_table;
		
		for (; refinement_level<max_refinement_level+1; ++refinement_level) {
			fracture_network.init_fractures(tria, n_qpoints);
			fracture_network.output_to_vtk("burman_frac");
			
			pressure_solver.run();
			
			velocity.setup_system();
			pressure_solver.calculate_velocity(velocity, true);
			pressure_solver.add_fracture_velocity(velocity, 1.0);
			
			const std::pair<double,double> velocity_error_CG = velocity.calculate_flux_error(rock, flow_fun.exact_gradient);
			residuals.reinit(tria.n_active_cells());
			const double residual_CG_L2 = velocity.calculate_residuals(residuals, flow_fun, fracture_network);
			
			if ( ! velocity.is_locally_conservative()) {
				postprocessor.setup_system();
				Vector<double> correction = postprocessor.apply(residuals);
				velocity.add_correction(correction);
			}
			const std::pair<double,double> velocity_error_PP = velocity.calculate_flux_error(rock, flow_fun.exact_gradient);
			
			conv_table.add_value("h", (exp(5.0/4.0)-1.0)/pow(2.0,refinement_level));
			conv_table.add_value("DoFs", dh_pressure.n_dofs());
			conv_table.add_value("||p-p_h||", pressure_solver.pressure_l2_error());
			conv_table.add_value("||grad(p-p_h)||", pressure_solver.gradp_l2_error());
			conv_table.add_value("||U-U_h||", velocity_error_CG.second);
			conv_table.add_value("||U-V_h||", velocity_error_PP.second);
			conv_table.add_value("R(U_h)", residual_CG_L2);
			
			tria.refine_global();
			rock.execute_coarsening_and_refinement();
		}
		
		conv_table.omit_column_from_convergence_rate_evaluation("h");
		conv_table.omit_column_from_convergence_rate_evaluation("DoFs");
		conv_table.evaluate_all_convergence_rates(ConvergenceTable::RateMode::reduction_rate_log2);
		conv_table.write_text(std::cout);
	}
	
	return 0;
}
