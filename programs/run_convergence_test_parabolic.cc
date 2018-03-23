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
#include "../source/ParabolicPressureSolver.h"
#include "../source/PostProcessGS.h"
#include "../source/PostProcessMM.h"
#include "../source/HelpFunctions.h"
#include "../source/TransportSolver.h"

/* Run a sequence of refinements and output error norms and convergence rates
 */
int main (int argc, char *argv[])
{
	deallog.depth_console(0);
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
	const int dim = 2;

	const ProblemType pt = ProblemType::SANGHYUN;
	const bool local_refinement = false;
	const double dirichlet_penalty = 100.0;
	const int n_qpoints = 2;
	const double residual_tolerance = 1e-10;
	double dt = 0.05;
	const double t_end = 0.1;
	const double alpha = 1.0;
	const bool weak_bcs = false;
	const bool use_dirichlet_flux = false;
	const bool harmonic_weighting = false;
	const bool weighted_norm = false;

	const int first_refinement = 2;
	const int last_refinement  = 4;

	// Create grid
	Triangulation<dim> triangulation;
	make_grid(triangulation, pt, first_refinement, local_refinement, 0.0);

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
			      << std::endl << "1/h = " << pow(2,refinement+local_refinement)
				  << std::endl << "------------------------------"
				  << std::endl;

		// Initialize pressure solver
		ParabolicPressureSolver<dim> pressure_solver(triangulation, fe_pressure, dh_pressure, rock, flow_fun);
		pressure_solver.set_parameters(dirichlet_penalty, "solution_pressure", n_qpoints, weak_bcs);
		pressure_solver.setup_system();
		if (use_dirichlet_flux)
			pressure_solver.setup_dirichlet_flux_recovery();

		// Initialize postprocessors
		PostProcessMM<dim> postprocessor(triangulation, fe_correction, dh_correction, rock);
		postprocessor.set_parameters(residual_tolerance, false, !use_dirichlet_flux, weighted_norm);
		postprocessor.setup_system();

		// Initialize velocity data
		VelocityData<dim> velocity_CG(triangulation);
		velocity_CG.setup_system();
		VelocityData<dim> velocity_MM(triangulation);
		velocity_MM.setup_system();
		VelocityData<dim> velocity_exact(triangulation);
		velocity_exact.setup_system();

		// Initialize transport solvers
		TransportSolver<dim> transport_solver_CG(triangulation, fe_transport, dh_transport, rock, transport_fun, pt);
		transport_solver_CG.set_runtime_parameters("solution_transport_CG", n_qpoints, 0.0);
		transport_solver_CG.setup_system();
		TransportSolver<dim> transport_solver_PP(triangulation, fe_transport, dh_transport, rock, transport_fun, pt);
		transport_solver_PP.set_runtime_parameters("solution_transport_PP", n_qpoints, 0.0);
		transport_solver_PP.setup_system();
		TransportSolver<dim> transport_solver_exact(triangulation, fe_transport, dh_transport, rock, transport_fun, pt);
		transport_solver_exact.set_runtime_parameters("solution_transport_exact", n_qpoints, 0.0);
		transport_solver_exact.setup_system();

		// Vectors to store solutions
		Vector<double> pressure, pressure_old, residuals, correction;
		residuals.reinit(triangulation.n_active_cells());

		double CG_residual_l2norm = NaN;

		// Time loop
		double time = dt;
		int    time_step = 1;
		while (time < (t_end + dt/100)) {
			// Pressure solver
			pressure_old = pressure_solver.get_pressure_solution();
			print_header("Pressure Solver");
			pressure_solver.solve_time_step(dt);
			pressure = pressure_solver.get_pressure_solution();
			if (use_dirichlet_flux)
				pressure_solver.apply_dirichlet_flux_recovery(dt);
			pressure_solver.calculate_velocity(velocity_CG, harmonic_weighting);

			// Calculate residuals
			print_header("Residual");
			residuals.reinit(triangulation.n_active_cells());
			flow_fun.set_time(time);
			Vector<double> pressure_diff = pressure;
			pressure_diff -= pressure_old;
			pressure_diff *= alpha/dt;
			CG_residual_l2norm = velocity_CG.calculate_residuals(residuals, flow_fun.right_hand_side,
																 &fe_pressure, &dh_pressure, pressure_diff);

			// Postprocessing
			print_header("Postprocessing");
			correction = postprocessor.apply(residuals);

			// Transport
			print_header("Transport Solver");
			transport_solver_CG.set_velocity(velocity_CG);
			transport_solver_CG.solve_time_step(dt);

			pressure_solver.calculate_velocity(velocity_MM, harmonic_weighting);
			velocity_MM.add_correction(correction);
			transport_solver_PP.set_velocity(velocity_MM);
			transport_solver_PP.solve_time_step(dt);

			velocity_exact.init_exact(flow_fun);
			transport_solver_exact.set_velocity(velocity_exact);
			transport_solver_exact.solve_time_step(dt);

			// Increment time
			time += dt;
			++time_step;
		}

		// Read error norms and add to convergence table
		std::pair<double,double> velocity_error = velocity_CG.calculate_flux_error(rock, flow_fun.exact_gradient);
		const double CG_flux_l2_norm  = velocity_error.first;
		const double CG_flux_edgenorm = velocity_error.second;
		const double CG_gradp_l2_norm = pressure_solver.gradp_l2_error();
		const double CG_pressure_l2_norm = pressure_solver.pressure_l2_error();
		
		const double PP_residual_l2norm = postprocessor.get_residual_l2norm();
		velocity_error = velocity_MM.calculate_flux_error(rock, flow_fun.exact_gradient);
		const double PP_flux_l2_error   = velocity_error.first;
		const double PP_flux_edge_error = velocity_error.second;
		const double PP_correction_l2_norm  = postprocessor.get_correction_l2_norm();
		const double PP_correction_edgenorm = postprocessor.get_correction_edgenorm();

		convergence_table_pressure.add_value("1/h", pow(2,refinement));
		convergence_table_pressure.add_value("dt", dt);
		convergence_table_pressure.add_value("p L2", CG_pressure_l2_norm);
		convergence_table_pressure.add_value("p H1", CG_gradp_l2_norm);
		convergence_table_pressure.add_value("U L2", CG_flux_l2_norm);
		convergence_table_pressure.add_value("U e-norm", CG_flux_edgenorm);
		convergence_table_pressure.add_value("V L2", PP_flux_l2_error);
		convergence_table_pressure.add_value("V e-norm", PP_flux_edge_error);
		convergence_table_pressure.add_value("CG R L2", CG_residual_l2norm);
		convergence_table_pressure.add_value("PP R L2", PP_residual_l2norm);
		convergence_table_pressure.add_value("PP G L2", PP_correction_l2_norm);
		convergence_table_pressure.add_value("PP G e-norm", PP_correction_edgenorm);

		const double CG_conc_l2_error = transport_solver_CG.get_l2_error_norm();
		const double CG_conc_min = transport_solver_CG.get_solution_min();
		const double CG_conc_max = transport_solver_CG.get_solution_max();
		const double PP_conc_l2_error = transport_solver_PP.get_l2_error_norm();
		const double PP_conc_min = transport_solver_PP.get_solution_min();
		const double PP_conc_max = transport_solver_PP.get_solution_max();

		const double EXACT_conc_l2_error = transport_solver_exact.get_l2_error_norm();
		const double EXACT_conc_min = transport_solver_exact.get_solution_min();
		const double EXACT_conc_max = transport_solver_exact.get_solution_max();

		convergence_table_transport.add_value("1/h", pow(2,refinement));
		convergence_table_transport.add_value("dt", dt);
		convergence_table_transport.add_value("c L2", EXACT_conc_l2_error);
		convergence_table_transport.add_value("CG: c L2", CG_conc_l2_error);
		convergence_table_transport.add_value("PP: c L2", PP_conc_l2_error);
		convergence_table_transport.add_value("c_min", EXACT_conc_min);
		convergence_table_transport.add_value("CG c_min", CG_conc_min);
		convergence_table_transport.add_value("PP c_min", PP_conc_min);
		convergence_table_transport.add_value("c_max", EXACT_conc_max);
		convergence_table_transport.add_value("CG c_max", CG_conc_max);
		convergence_table_transport.add_value("PP c_max", PP_conc_max);

		// Refine grid and decrease time step
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
	convergence_table_pressure.set_precision("PP R L2", 3);
	convergence_table_pressure.set_scientific("PP R L2", true);
	convergence_table_pressure.set_precision("PP G L2", 6);
	convergence_table_pressure.set_precision("PP G e-norm", 6);

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
	convergence_table_transport.set_precision("PP: c L2", 5);
	convergence_table_transport.set_precision("c_min", 5);
	convergence_table_transport.set_precision("c_max", 5);
	convergence_table_transport.set_precision("CG c_min", 5);
	convergence_table_transport.set_precision("CG c_max", 5);
	convergence_table_transport.set_precision("PP c_min", 5);
	convergence_table_transport.set_precision("PP c_max", 5);

	convergence_table_transport.omit_column_from_convergence_rate_evaluation("1/h");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("dt");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("c_min");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("c_max");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("CG c_min");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("CG c_max");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("PP c_min");
	convergence_table_transport.omit_column_from_convergence_rate_evaluation("PP c_max");
	convergence_table_transport.evaluate_all_convergence_rates(ConvergenceTable::RateMode::reduction_rate_log2);

	convergence_table_transport.write_text(std::cout);
	std::ofstream ctt_file("convergence_transport.tex");
	convergence_table_transport.write_tex(ctt_file, false);
	ctt_file.close();

	std::cout << std::endl;

	return 0;
}
