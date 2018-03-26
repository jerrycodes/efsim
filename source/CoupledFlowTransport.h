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

#ifndef COUPLED_FLOW_TRANSPORT_H
#define COUPLED_FLOW_TRANSPORT_H


#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>

#include "ParabolicPressureSolver.h"
#include "EllipticPressureSolver.h"
#include "PostProcessGS.h"
#include "PostProcessMM.h"
#include "HelpFunctions.h"
#include "TransportSolver.h"
#include "RockProperties.h"
#include "VTKReader.h"
#include "ProblemFunctions.h"
#include "RockProperties.h"
#include "Velocity.h"

using namespace dealii;


/* Main driver class.
 * Reads parameters from command line and run coupled flow problem:
 * - Pressure solver
 * - Postprocessing (optional)
 * - Transport solver
 */
template <int dim>
class CoupledFlowTransport
{
public:
	CoupledFlowTransport();
	void read_parameters(ParameterHandler& p);
	void run();

private:
	void declare_default_parameters();
	void setup();

	Triangulation<dim> triangulation;

	// Pressure variables: Q1
	FE_Q<dim>       fe_pressure;
	DoFHandler<dim> dh_pressure;

	// Flux correction variables: Face constants
	FE_FaceQ<dim>   fe_correction;
	DoFHandler<dim> dh_correction;

	// Transport variables: DG0
	FE_DGQ<dim>     fe_transport;
	DoFHandler<dim> dh_transport;

	enum PostProcessingMethod {MM, GS, NONE};

	RockProperties<dim> rock;
	ProblemFunctionsFlow<dim> flow_fun;
	ProblemFunctionsTransport<dim> transport_fun;

	// Input parameters
	ParameterHandler* param;
	ProblemType problem_type;
	double dt;
	double dt2;
	double t_change_dt;
	double t_end;
	PostProcessingMethod pp_method;
	double dirichlet_penalty;
	double alpha;
	bool use_dirichlet_flux;
	bool do_transport;
	bool harmonic_average;
	
	// TODO: Take these as input
	double frac_perm = 1e4;
	double frac_width = 1e-4;
};


/* Constructor
 * Initialize finite elements and dof handlers.
 */
template <int dim>
CoupledFlowTransport<dim>::CoupledFlowTransport()
: fe_pressure(1),
  dh_pressure(triangulation),
  fe_correction(0),
  dh_correction(triangulation),
  fe_transport(0),
  dh_transport(triangulation),
  rock(triangulation),
  flow_fun(),
  transport_fun()
{ }


/* Read parameters from command line and store in ParameterHandler param.
 * Also extract member variables.
 */
template <int dim>
void CoupledFlowTransport<dim>::read_parameters(ParameterHandler& p)
{
	param = &p;

	// Read parameter file from command line
	print_header("READ PARAMETERS");
	param->print_parameters(std::cout, ParameterHandler::OutputStyle::ShortText);

	// Retrieve some parameters
	problem_type = getProblemType(*param, "Problem type", "Global");
	param->enter_subsection("Global");
	dt = param->get_double("Time step size");
	dt2 = param->get_double("Time step size 2");
	t_change_dt = param->get_double("Time to change dt");
	t_end = param->get_double("End time");
	param->leave_subsection();
	param->enter_subsection("Pressure solver");
	dirichlet_penalty = param->get_double("Dirichlet penalty");
	alpha = param->get_double("Alpha");
	param->leave_subsection();
	param->enter_subsection("Transport solver");
	do_transport = param->get_bool("Do transport");
	param->leave_subsection();

	// Get postprocessing method
	param->enter_subsection("Postprocessing");
	std::string method = param->get("Method");
	use_dirichlet_flux = param->get_bool("Dirichlet flux recovery");
	harmonic_average   = param->get_bool("Harmonic weighting");
	param->leave_subsection();
	if (method == "GS")        pp_method = PostProcessingMethod::GS;
	else if (method == "MM")   pp_method = PostProcessingMethod::MM;
	else if (method == "NONE") pp_method = PostProcessingMethod::NONE;
	else {
		std::cout << "Warning: Parameter Postprocessing::Method unknown."
		    	  << "Using MM method." << std::endl;
		pp_method = PostProcessingMethod::MM;
	}
}


// Create grid, distribute dofs and get rock properties
template <int dim>
void CoupledFlowTransport<dim>::setup()
{
	param->enter_subsection("Grid");
	std::string gt = param->get("Grid type");
	if (gt == "1D") {
		make_grid_1D(triangulation, param->get_integer("Global refinement"), param->get_bool("Uniform 1D grid"));
	}
	else if (gt == "VTK") {
		Assert(dim == 3, ExcInternalError());
		problem_type = VTK;
		const std::string vtkfilename = param->get("VTK file");
		std::ifstream vtkfile(vtkfilename);
		if ( ! vtkfile.is_open()) {
			std::cout << "Error: Not able to read input file." << std::endl;
			std::cout << "Usage: ./read_cpgrid gridfile.vtk" << std::endl;
			exit(1);
		}
		VTKReader<dim> vtk_reader;
		vtk_reader.attach_triangulation(triangulation);
		vtk_reader.read_vtk(vtkfile);
		vtkfile.close();
		rock.initialize(vtk_reader.get_poro(), vtk_reader.get_perm());
		set_linear_pressure_bcs(triangulation, flow_fun, 1);
		//set_well_bcs(triangulation, flow_fun, transport_fun);
	}
	else {
		// TODO: Use get_boundary_id(ProblemType) instead of declaring boundaries in make_grid functions
		if (problem_type == REGULAR_NETWORK) {
			if (param->get_bool("Do local refinement")) {
				make_grid_regular_network(triangulation, 16, true);
				FractureNetwork fractures_tmp(REGULAR_NETWORK);
				refine_around_fractures(triangulation, fractures_tmp, 4);
			}
			else
				make_grid_regular_network(triangulation, 37);
		}
		else if (problem_type == REGULAR_NETWORK_RESOLVED)
			make_grid_regular_network_resolved(triangulation);
		else if (problem_type == SIMPLE_FRAC_RESOLVED)
			make_grid_single_frac(triangulation);
		else
			make_grid(triangulation, problem_type,
					  param->get_integer("Global refinement"),
					  param->get_bool("Do local refinement"),
					  param->get_double("Distortion factor"));
	}
	param->leave_subsection();

	if ( gt != "VTK" ) {
		rock.initialize(problem_type);
		flow_fun.set_problem(problem_type);
		transport_fun.set_problem(problem_type);
	}
	
	// Output grid to eps file	
	const std::string gridfile = "grid.eps";
	std::ofstream outstream(gridfile);
	GridOut grid_out;
	grid_out.write_eps(triangulation, outstream);
}


/* Main class driver
 * Sets up and initialize solver and postprocessing classes.
 * Then run pressure solver, postprocessing and transport solver.
 */
template <int dim>
void CoupledFlowTransport<dim>::run() {
	setup();

	VelocityData<dim> velocity(triangulation);
	velocity.setup_system();
	
	FractureNetwork fracture_network(problem_type);
	Assert(dim == 2 || fracture_network.n_fractures() == 0, ExcMessage("Assuming no fractures for dim!=2."));

	// Initialize pressure solver
	// If alpha = 0, choose EllipticPressureSolver, or else ParabolicPressureSolver
	PressureSolverBase<dim>* pressure_solver;
	if (alpha == 0) {
		pressure_solver = new EllipticPressureSolver<dim>(triangulation, fe_pressure, dh_pressure, rock, flow_fun);
		pressure_solver->set_fractures(fracture_network, frac_perm, frac_width);
	}
	else
		pressure_solver = new ParabolicPressureSolver<dim>(triangulation, fe_pressure, dh_pressure, rock, flow_fun);
		
	pressure_solver->set_parameters(*param);
	pressure_solver->setup_system();
	if (use_dirichlet_flux)
		pressure_solver->setup_dirichlet_flux_recovery();

	// Initialize postprocessor
	PostProcessBase<dim>* postprocessor;
	if (pp_method == PostProcessingMethod::GS)
		postprocessor = new PostProcessGS<dim>(triangulation, fe_correction, dh_correction);
	else if (pp_method == PostProcessingMethod::MM) {
		postprocessor = new PostProcessMM<dim>(triangulation, fe_correction, dh_correction, rock);
		postprocessor->set_fractures(frac_perm);
	}
    else {
		Assert(pp_method == PostProcessingMethod::NONE, ExcInternalError());
		postprocessor = NULL;
	}

	if (pp_method != PostProcessingMethod::NONE) {
		postprocessor->set_parameters(*param);
	}

	// Intialize transport solver
	TransportSolver<dim> transport_solver(triangulation, fe_transport, dh_transport, rock, transport_fun, problem_type);

	if (do_transport) {
		transport_solver.set_runtime_parameters(*param);
		transport_solver.setup_system();
		transport_solver.set_fractures(fracture_network, frac_width);
	}
	
	fracture_network.init_fractures(triangulation);
	fracture_network.output_to_vtk("fracture");

	// Vectors to store solutions
	Vector<double> pressure, pressure_old, residuals, residuals_new, correction, concentration, dirichlet_flux;

	// If alpha = 0 (time-independent problem) solve for pressure before time loop
	if (alpha == 0) {
		print_header("Elliptic Pressure Solver");
		pressure_solver->run();
		pressure_solver->print_timing();
		pressure = pressure_solver->get_pressure_solution();
		if (use_dirichlet_flux)
			pressure_solver->apply_dirichlet_flux_recovery();
		pressure_solver->output_fracture_pressure();
		pressure_solver->calculate_velocity(velocity, harmonic_average);
		pressure_solver->add_fracture_velocity(velocity, frac_width);
		std::pair<double,double> flux_error = velocity.calculate_flux_error(rock, flow_fun.exact_gradient);
		std::cout << "L2 error of pressure:   " << pressure_solver->pressure_l2_error() << std::endl;
		std::cout << "L2 norm of gradp:       " << pressure_solver->gradp_l2_error() << std::endl;
		std::cout << "L2 norm of flux error:  " << flux_error.first << std::endl;
		std::cout << "Edgenorm of flux error: " << flux_error.second << std::endl;
	}

	// Time loop
	double time = dt;
	int    time_step = 1;
	while (time < (t_end + dt/100)) {
		std::cout << std::endl
				  << "------------------------------------" << std::endl
				  << " Time step " << time_step << " ( t = " << time << "s)" << std::endl
				  << "------------------------------------" << std::endl;

		pressure_old = pressure_solver->get_pressure_solution();

		// Run pressure solver if alpha != 0
		if (alpha != 0.0) {
			flow_fun.set_time(time);
			print_header("Parabolic Pressure Solver");
			pressure_solver->solve_time_step(dt);
			pressure = pressure_solver->get_pressure_solution();
			pressure_solver->print_timing();
			if (use_dirichlet_flux)
				pressure_solver->apply_dirichlet_flux_recovery(dt);
			pressure_solver->calculate_velocity(velocity, harmonic_average);
			pressure_solver->add_fracture_velocity(velocity, frac_width);
			std::pair<double,double> flux_error = velocity.calculate_flux_error(rock, flow_fun.exact_gradient);
			std::cout << "L2 error of pressure:   " << pressure_solver->pressure_l2_error() << std::endl;
			std::cout << "L2 norm of gradp:       " << pressure_solver->gradp_l2_error() << std::endl;
			std::cout << "L2 norm of flux error:  " << flux_error.first << std::endl;
			std::cout << "Edgenorm of flux error: " << flux_error.second << std::endl;
		}

		// Calculate residuals and do postprocessing if time-dependent problem or if at first time step
		if (alpha != 0.0 || time_step == 1) {
			// Residual calculations
			print_header("Residual");
			residuals.reinit(triangulation.n_active_cells());
			Vector<double> pressure_diff_scaled = pressure;
			pressure_diff_scaled -= pressure_old;
			pressure_diff_scaled *= alpha/dt;
			//double residual_l2norm = velocity.calculate_residuals(residuals, flow_fun.right_hand_side,
			//														&fe_pressure, &dh_pressure, pressure_diff_scaled);
			double residual_l2norm = velocity.calculate_residuals(residuals, flow_fun, fracture_network);
			
			std::cout << "L2 norm of residual: " << residual_l2norm << std::endl;

			velocity.write_to_vtk("velocity_CG");

			// Perform postprocessing if needed and wanted
			if ( (! velocity.is_locally_conservative()) && ( pp_method != PostProcessingMethod::NONE) ) {
				print_header("Postprocessing");
				postprocessor->setup_system();
				correction = postprocessor->apply(residuals);
				postprocessor->print_timing();
				std::cout << "L2 norm of residual: " << postprocessor->get_residual_l2norm() << std::endl;
				velocity.add_correction(correction);
				std::pair<double,double> flux_error = velocity.calculate_flux_error(rock, flow_fun.exact_gradient);
				std::cout << "L2 norm of flux error:   " << flux_error.first << std::endl;
				std::cout << "Edge norm of flux error: " << flux_error.second << std::endl;
				
				velocity.write_to_vtk("velocity_PP");
				velocity.output_fracture_velocity(&fracture_network, frac_width);
			}
			else {
				correction.reinit(dh_correction.n_dofs());
			}
			
			// Send pressure solution to transport solver
			if (do_transport)
				transport_solver.set_velocity(velocity);
		}

		// Run transport
		if (do_transport) {
			print_header("Transport Solver");
			transport_solver.solve_time_step(dt);
		}

		// Increment time
		if (time >= t_change_dt)
			dt = dt2;
		time += dt;
		++time_step;
	}

	// Delete object pointed to by pressure solver
	delete pressure_solver;

	// Delete object pointed to by postprocessor
	// Only if a proper postprocessor is pointed to
	if (pp_method != PostProcessingMethod::NONE)
		delete postprocessor;
}


#endif // COUPLED_FLOW_TRANSPORT_H
