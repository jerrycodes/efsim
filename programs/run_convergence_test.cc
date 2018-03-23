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

#include <deal.II/base/convergence_table.h>

#include "../source/ProblemFunctions.h"
#include "../source/EllipticPressureSolver.h"
#include "../source/PostProcessGS.h"
#include "../source/PostProcessMM.h"
#include "../source/HelpFunctions.h"
#include "../source/TransportSolver.h"
#include "../source/Velocity.h"

/* Run a sequence of refinements and output error norms and convergence rates
 */
int main (int argc, char *argv[])
{
	deallog.depth_console(0);
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
	const int dim = 2;

	const ProblemType pt = ProblemType::SANGHYUN_STEADYP;
	const bool local_refinement = false;
	const double dirichlet_penalty = 100.0;
	const int n_qpoints = 2;
	const double residual_tolerance = 1e-10;
	double dt = 0.05;
	const double t_end = 0.1;
	const bool weak_bcs = false;
	const bool use_dirichlet_flux = false;
	const bool harmonic_weighting = false;
	const bool weighted_norm = false;

	const int first_refinement = 2;
	const int last_refinement  = 5;

	// Create grid
	Triangulation<dim> triangulation;
	make_grid(triangulation, pt, first_refinement, local_refinement, 0.0);
	//make_grid_1D(triangulation, 0.25);
	//make_grid_2D(triangulation, 0.25);

	RockProperties<dim> rock(triangulation);
	rock.initialize(pt);
	ProblemFunctionsFlow<dim> flow_fun(pt);
	ProblemFunctionsRock<dim> rock_fun(pt);
	ProblemFunctionsTransport<dim> transport_fun(pt);

	// fe and dofs
	const FE_Q<dim>     fe_pressure(1);
	DoFHandler<dim>     dh_pressure(triangulation);
	const FE_FaceQ<dim> fe_correction(0);
	DoFHandler<dim>     dh_correction(triangulation);
	const FE_DGQ<dim>   fe_transport(0);
	DoFHandler<dim>		dh_transport(triangulation);

	ConvergenceTable convergence_table_pressure;
	ConvergenceTable convergence_table_transport;

	for (int refinement=first_refinement; refinement<(last_refinement+1); ++refinement) {
		std::cout << std::endl << "------------------------------"
			      << std::endl << "1/h = " << pow(2,refinement)
				  << std::endl << "------------------------------"
				  << std::endl;

		// Initialize pressure solver
		EllipticPressureSolver<dim> pressure_solver(triangulation, fe_pressure, dh_pressure, rock, flow_fun);
		std::string filename_p = "solution_pressure-";
		filename_p += ('0' + refinement);
		pressure_solver.set_parameters(dirichlet_penalty, filename_p, n_qpoints, weak_bcs);

		// Initialize postprocessors
		PostProcessMM<dim> postprocessorMM(triangulation, fe_correction, dh_correction, rock);
		PostProcessGS<dim> postprocessorGS(triangulation, fe_correction, dh_correction);
		postprocessorMM.set_parameters(residual_tolerance, false, !use_dirichlet_flux, weighted_norm);
		postprocessorGS.set_parameters(residual_tolerance, false, !use_dirichlet_flux);
		
		VelocityData<dim> velocity(triangulation);
		VelocityData<dim> velocity_exact(triangulation);
		
		// Initialize transport solver
		TransportSolver<dim> transport_solver(triangulation, fe_transport, dh_transport, rock, transport_fun, pt);
		std::string filename_t = "solution_transport-";
		filename_t += ('0' + refinement);
		transport_solver.set_runtime_parameters(filename_t, n_qpoints, 0.0);

		// Vectors to store solutions
		Vector<double> pressure, residuals, correctionMM, correctionGS;
		residuals.reinit(triangulation.n_active_cells());

		// Run pressure solver and calculate residual
		pressure_solver.run();
		pressure = pressure_solver.get_pressure_solution();
		if (use_dirichlet_flux) {
			pressure_solver.setup_dirichlet_flux_recovery();
			pressure_solver.apply_dirichlet_flux_recovery();
		}

		// Initialize velocity data from pressure solution
		velocity.setup_system();
		pressure_solver.calculate_velocity(velocity, harmonic_weighting);

		std::pair<double,double> velocity_error = velocity.calculate_flux_error(rock, flow_fun.exact_gradient);
		const double CG_flux_l2_norm = velocity_error.first;
		const double CG_flux_edgenorm = velocity_error.second;
		const double CG_gradp_l2_norm = pressure_solver.gradp_l2_error();
		const double CG_pressure_l2_norm = pressure_solver.pressure_l2_error();
		const double CG_residual_l2norm = velocity.calculate_residuals(residuals, flow_fun.right_hand_side);

		// Transport solver - CG flux
		transport_solver.run(dt, t_end, velocity);
		const double NONE_conc_l2_error = transport_solver.get_l2_error_norm();
		const double NONE_conc_min = transport_solver.get_solution_min();
		const double NONE_conc_max = transport_solver.get_solution_max();

		// Apply postprocessor
		postprocessorMM.setup_system();
		postprocessorGS.setup_system();
		correctionMM = postprocessorMM.apply(residuals);
		const double MM_residual_l2norm = postprocessorMM.get_residual_l2norm();
		velocity.add_correction(correctionMM);

		velocity_error = velocity.calculate_flux_error(rock, flow_fun.exact_gradient);
		const double MM_flux_l2_error = velocity_error.first;
		const double MM_flux_edge_error = velocity_error.second;
		const double MM_correction_l2_norm  = postprocessorMM.get_correction_l2_norm();
		const double MM_correction_edgenorm = postprocessorMM.get_correction_edgenorm();

		// Transport solver - MM flux
		transport_solver.run(dt, t_end, velocity);
		const double MM_conc_l2_error = transport_solver.get_l2_error_norm();
		const double MM_conc_min = transport_solver.get_solution_min();
		const double MM_conc_max = transport_solver.get_solution_max();

		// Transport solver - exact flux
		velocity_exact.setup_system();
		velocity_exact.init_exact(flow_fun);
		transport_solver.run(dt, t_end, velocity_exact);
		const double EXACT_conc_l2_error = transport_solver.get_l2_error_norm();
		const double EXACT_conc_min = transport_solver.get_solution_min();
		const double EXACT_conc_max = transport_solver.get_solution_max();

		convergence_table_pressure.add_value("1/h", pow(2,refinement));
		convergence_table_pressure.add_value("dt", dt);
		convergence_table_pressure.add_value("p L2", CG_pressure_l2_norm);
		convergence_table_pressure.add_value("p H1", CG_gradp_l2_norm);
		convergence_table_pressure.add_value("U L2", CG_flux_l2_norm);
		convergence_table_pressure.add_value("U e-norm", CG_flux_edgenorm);
		convergence_table_pressure.add_value("V L2", MM_flux_l2_error);
		convergence_table_pressure.add_value("V e-norm", MM_flux_edge_error);
		convergence_table_pressure.add_value("CG R L2", CG_residual_l2norm);
		convergence_table_pressure.add_value("MM R L2", MM_residual_l2norm);
		convergence_table_pressure.add_value("MM G L2", MM_correction_l2_norm);
		convergence_table_pressure.add_value("MM G e-norm", MM_correction_edgenorm);

		convergence_table_transport.add_value("1/h", pow(2,refinement));
		convergence_table_transport.add_value("dt", dt);
		convergence_table_transport.add_value("c L2", EXACT_conc_l2_error);
		convergence_table_transport.add_value("CG: c L2", NONE_conc_l2_error);
		convergence_table_transport.add_value("MM: c L2", MM_conc_l2_error);
		convergence_table_transport.add_value("c_min", EXACT_conc_min);
		convergence_table_transport.add_value("CG c_min", NONE_conc_min);
		convergence_table_transport.add_value("MM c_min", MM_conc_min);
		convergence_table_transport.add_value("c_max", EXACT_conc_max);
		convergence_table_transport.add_value("CG c_max", NONE_conc_max);
		convergence_table_transport.add_value("MM c_max", MM_conc_max);

		triangulation.refine_global();
		rock.execute_coarsening_and_refinement();
		dt = dt/4.0;
	}

	std::cout << std::endl;

	convergence_table_pressure.set_precision("1/h", 0);
	convergence_table_pressure.set_precision("dt", 5);
	convergence_table_pressure.set_precision("p L2", 6);
	convergence_table_pressure.set_precision("p H1", 6);
	convergence_table_pressure.set_precision("U L2", 6);
	convergence_table_pressure.set_precision("U e-norm", 6);
	convergence_table_pressure.set_precision("V L2", 6);
	convergence_table_pressure.set_precision("V e-norm", 6);
	convergence_table_pressure.set_precision("CG R L2", 6);
	convergence_table_pressure.set_precision("MM R L2", 3);
	convergence_table_pressure.set_scientific("MM R L2", true);
	convergence_table_pressure.set_precision("MM G L2", 6);
	convergence_table_pressure.set_precision("MM G e-norm", 6);

	convergence_table_pressure.omit_column_from_convergence_rate_evaluation("1/h");
	convergence_table_pressure.omit_column_from_convergence_rate_evaluation("dt");
	convergence_table_pressure.evaluate_all_convergence_rates(ConvergenceTable::RateMode::reduction_rate_log2);

	convergence_table_pressure.write_text(std::cout);
	std::ofstream ctp_file("convergence_pressure.tex");
	convergence_table_pressure.write_tex(ctp_file, false);
	ctp_file.close();

	std::cout << std::endl;

	convergence_table_transport.set_precision("1/h", 0);
	convergence_table_transport.set_precision("dt", 5);
	convergence_table_transport.set_precision("c L2", 5);
	convergence_table_transport.set_precision("CG: c L2", 5);
	convergence_table_transport.set_precision("MM: c L2", 5);
	convergence_table_transport.set_precision("c_min", 5);
	convergence_table_transport.set_precision("c_max", 5);
	convergence_table_transport.set_precision("CG c_min", 5);
	convergence_table_transport.set_precision("CG c_max", 5);
	convergence_table_transport.set_precision("MM c_min", 5);
	convergence_table_transport.set_precision("MM c_max", 5);

	convergence_table_transport.omit_column_from_convergence_rate_evaluation("1/h");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("dt");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("c_min");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("c_max");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("CG c_min");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("CG c_max");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("MM c_min");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("MM c_max");
	convergence_table_transport.evaluate_all_convergence_rates(ConvergenceTable::RateMode::reduction_rate_log2);

	convergence_table_transport.write_text(std::cout);
	std::ofstream ctt_file("convergence_transport.tex");
	convergence_table_transport.write_tex(ctt_file, false);
	ctt_file.close();

	std::cout << std::endl;

	return 0;
}
