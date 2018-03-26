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
 
#include "../source/TransportSolver.h"
#include "../source/HelpFunctions.h"
#include "../source/RockProperties.h"
#include "../source/EllipticPressureSolver.h"
#include "../source/PostProcessMM.h"
#include "../source/Velocity.h"

#include <deal.II/base/convergence_table.h>

 
using namespace dealii;


// Main program for fractured media


unsigned int read_parameters(int argc, char *argv[], ParameterHandler& param);
void declare_parameters_fracture(ParameterHandler& param);


// Solve coupled flow and transport problem
int main (int argc, char *argv[])
{
	deallog.depth_console(2);
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
	
	ParameterHandler param;
	const unsigned int dim = 2; 
	const unsigned int dim_input = read_parameters(argc, argv, param);
	Assert(dim_input == dim, ExcMessage("Program only implemented for space dimension equal to 2."));
	
	// Get input data
	param.enter_subsection("Grid");
	const unsigned int Nx = param.get_integer("Nx");
	const unsigned int Ny = param.get_integer("Ny");
	const unsigned int global_ref = param.get_integer("Global refinement");
	const unsigned int n_ref_around_fractures = param.get_integer("Refine around fractures");
	const bool resolve_fractures = param.get_bool("Resolve close fractures");
	param.leave_subsection();
	param.enter_subsection("Global");
	const ProblemType pt = getProblemType(param, "Problem type");
	const double dt = param.get_double("Time step size");
	const double T = param.get_double("End time");
	const unsigned int n_qpoints = param.get_integer("No quadrature points");
	const bool use_explicit_velocity = param.get_bool("Use explicit velocity");
	param.leave_subsection();
	param.enter_subsection("Fracture");
	const double perm_frac = param.get_double("Permeability fracture");
	const double fracture_width = param.get_double("Width");
	const double velocity_frac = param.get_double("Velocity fracture");
	const string fracture_filename = param.get("Output file base");
	param.leave_subsection();
	param.enter_subsection("Postprocessing");
	const bool apply_postprocessing = param.get_bool("Apply postprocessing");
	const bool harmonic_weighting = param.get_bool("Harmonic weighting");
	param.leave_subsection();
	param.enter_subsection("Transport solver");
	const double do_transport = param.get_bool("Do transport");
	const std::string ref_sol_filename = param.get("Reference solution file");
	param.leave_subsection();
	
	
	// Make grid
	Triangulation<dim> tria;
	if (pt == FLEMISCH) {
		make_grid_flemisch(tria, Nx, n_ref_around_fractures>0);
	}
	else if (pt == FLEMISCH_RESOLVED) {
		make_grid_flemisch_resolved(tria);
	}
	else if (pt == SIMPLE_FRAC_RESOLVED)
		make_grid_single_frac(tria);
	else if (pt == COMPLEX_NETWORK) {
		make_grid_network(tria, global_ref);
	}
	else {
		std::vector<unsigned int> rep(2, Nx);
		rep[1] = Ny;
		Point<dim> p1, p2;
		p2[0] = 1.0; p2[1] = 1.0;
		GridGenerator::subdivided_hyper_rectangle(tria, rep, p1, p2);
		
		// Set pressure boundary conditions
		std::vector<unsigned int> boundary_ids = get_boundary_id(pt);
		for (typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active(); cell != tria.end(); ++cell) {
			for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f) {
				if (cell->at_boundary(f))
					cell->face(f)->set_boundary_id(boundary_ids[f]);
			}
		}
	}
	
	// Set up fracture network
	FractureNetwork fracture_network(pt);
	refine_around_fractures(tria, fracture_network, n_ref_around_fractures);
	if (resolve_fractures)
		refine_around_close_fractures(tria, fracture_network);
	
	// Finite elements and dof handler for pressure and correction
	FE_Q<dim> fe_pressure(1);
	DoFHandler<dim> dh_pressure(tria);
	FE_FaceQ<dim> fe_correction(0);
	DoFHandler<dim> dh_correction(tria);
	FE_DGQ<dim> fe_conc(0);
	DoFHandler<dim> dh_conc(tria);
	
	// Problem data
	RockProperties<dim> rock(tria);
	ProblemFunctionsFlow<dim> flow_fun(pt);
	ProblemFunctionsTransport<dim> transport_fun(pt);
	if (pt == COMPLEX_NETWORK) {
		rock.initialize(1.0, 1e-14);
		flow_fun.set_linear_pressure(0, tria, 1013250);
	}
	else
		rock.initialize(pt);
	
	// Initialize elliptic pressure solver
	EllipticPressureSolver<dim> pressure_solver(tria, fe_pressure, dh_pressure, rock, flow_fun);
	pressure_solver.set_parameters(param);
	pressure_solver.set_fractures(fracture_network, perm_frac, fracture_width);
	
	// Intialize postprocessor
	PostProcessMM<dim> postprocessor(tria, fe_correction, dh_correction, rock);
	postprocessor.set_parameters(param);
	postprocessor.set_fractures(perm_frac);
	
	// Initialize velocity handler
	VelocityData<dim> velocity(tria);
	
	// Vectors to hold residuals and correction from postprocessor
	Vector<double> residuals, correction;
	
	// Intialize transport solver
	TransportSolver<dim> transport_solver(tria, fe_conc, dh_conc, rock, transport_fun, pt);
	transport_solver.set_runtime_parameters(param);
	transport_solver.set_fractures(fracture_network, fracture_width);
	
	
	fracture_network.init_fractures(tria, n_qpoints);
	fracture_network.output_to_vtk(fracture_filename);
	
	velocity.setup_system();
	
	if (!use_explicit_velocity) {
		print_header("Pressure solver");
		pressure_solver.run();
		
		std::cout << "L2 error pressure: " << pressure_solver.pressure_l2_error() << std::endl;
		
		print_header("Velocity and residual calculations");
		pressure_solver.calculate_velocity(velocity, harmonic_weighting);
		pressure_solver.add_fracture_velocity(velocity, fracture_width);
		
		residuals.reinit(tria.n_active_cells());
		velocity.calculate_residuals(residuals, flow_fun, fracture_network);
		velocity.write_to_vtk("velocity_CG");
		
		if ( apply_postprocessing && (! velocity.is_locally_conservative()) ) {
			print_header("Postprocessing");
			postprocessor.setup_system();
			Vector<double> correction = postprocessor.apply(residuals);
			std::cout << "L2 norm of residual: " << postprocessor.get_residual_l2norm() << std::endl;
			velocity.add_correction(correction);
		}
		velocity.write_to_vtk("velocity_PP");
		velocity.output_fracture_velocity(&fracture_network, 1e-4);
	}
	else  {
		velocity.init_exact(flow_fun);
		velocity.set_constant_fracture_velocity(fracture_network, velocity_frac);
		velocity.write_to_vtk("velocity_explicit");
	}
	
	if (do_transport) {
		print_header("Transport Solver (DG(0) + IE)");
		transport_solver.run(dt, T, velocity);
		
		if (pt == FLEMISCH_RESOLVED) {
			FractureNetwork fractures_tmp(FLEMISCH);
			fractures_tmp.init_fractures(tria);
			transport_solver.set_fractures(fractures_tmp, fracture_width);
			transport_solver.output_fracture_solution();
			transport_solver.set_fractures(fracture_network, fracture_width);
		}
	}
	
	// Compare to a reference solution
	if (! ref_sol_filename.empty())
		std::cout << "L2 error of conc (by reference solution): " << transport_solver.l2_error_reference_solution(ref_sol_filename) << std::endl;;
	
	return 0;
}


unsigned int read_parameters(int argc, char *argv[], ParameterHandler& param)
{
	// Declare parameters and read input file
	declare_parameters_fracture(param);
	
	if ( argc > 1 && std::string(argv[1]) == "-help") {
		std::cout << "Copyright (C) 2018  Lars Hov Odsæter" << std::endl
			      << "Open license, GNU GPLv3: https://www.gnu.org/copyleft/gpl.html" << std::endl << std::endl;
		std::cout << "Usage: " << argv[0] << " input.prm" << std::endl << std::endl;
		param.print_parameters(std::cout, ParameterHandler::OutputStyle::Text);
		exit(0);
	}
	
	if (argc < 2)
		std::cout << "Warning: No input given. Using default setup."
				  << std::endl << std::endl;
	else {
		char* parameter_file = argv[1];
		if ( ! param.read_input(parameter_file, true) ) {
			std::cout << "  -> Using default setup." << std::endl << std::endl;
		}
		if (argc > 2)
			std::cout << "Warning: More than one input argument given. Ignoring all but the first."
					  << std::endl << std::endl;
	}
	
	// Read and return dimension
	param.enter_subsection("Grid");
	unsigned int dim = param.get_integer("Dimension");
	param.leave_subsection();
	
	return dim;
}


void declare_parameters_fracture(ParameterHandler& param)
{
	param.enter_subsection("Grid");
	{
		param.declare_entry("Dimension", 
							"2", 
							Patterns::Integer(), 
							"Problem dimension");
		param.declare_entry("Nx",
						    "10",
							Patterns::Integer(),
							"Number of elements x-direction");
		param.declare_entry("Ny",
						    "10",
							Patterns::Integer(),
							"Number of elements y-direction");
		param.declare_entry("Global refinement",
						    "4",
							Patterns::Integer(),
							"Number of global refinements");
		param.declare_entry("Refine around fractures",
							"0",
							Patterns::Integer(),
							"Number of refinement steps around fractures");
		param.declare_entry("Resolve close fractures",
							"true",
							Patterns::Bool(),
							"Resolve fractures that almost intersect");
	}
	param.leave_subsection();
	param.enter_subsection("Global");
	{
		param.declare_entry("Problem type",
						    "FRACTURE_TEST",
							Patterns::Anything(),
							"Problem type, see ProblemFunctions.h for list of possible choices");
		param.declare_entry("Time step size",
							"0.01",
							Patterns::Double(),
							"Time step size");
		param.declare_entry("End time",
							"1.0",
							Patterns::Double(),
							"End time for simulations");
		param.declare_entry("No quadrature points",
				            "2",
							Patterns::Double(),
							"Number of quadrature points");
		param.declare_entry("Use explicit velocity",
							"false",
							Patterns::Bool(),
							"Use explicit velocity in transport solver");
	}
	param.leave_subsection();
	param.enter_subsection("Fracture");
	{
		param.declare_entry("Permeability fracture",
							"1.0",
							Patterns::Double(),
							"Permeability in fractures");
		param.declare_entry("Width",
							"1.0",
							Patterns::Double(),
							"Fracture width");
		param.declare_entry("Velocity fracture",
							"0.0",
							Patterns::Double(),
							"Fracture velocity (if use explicit velocity)");
		param.declare_entry("Output file base",
							"fracture_filename",
							Patterns::Anything(),
							"Where to store fracture path output");
	}
	param.leave_subsection();
	param.enter_subsection("Pressure solver");
	{
		param.declare_entry("Dirichlet penalty",
						    "10.0",
							Patterns::Double(),
							"Dirichlet penalty coefficient (if weak boundary conditions)");
		param.declare_entry("Output file base",
							"solution_pressure",
							Patterns::Anything(),
							"Where to store results from pressure solver");
		param.declare_entry("S form",
							"1.0",
							Patterns::Double(),
							"S form = -1.0 for NIPG or OBB-DG; = 1.0 for SIPG; = 0.0 for IIPG");
		param.declare_entry("Weak BCs",
							"false",
							Patterns::Bool(),
							"Use weak Dirichlet boundary conditions or not");
		param.declare_entry("Linear solver tolerance",
							"1e-10",
							Patterns::Double(),
							"Linear solver tolerance");
		param.declare_entry("SSOR relaxation coeff",
							"1.5",
							Patterns::Double(),
							"Relaxation coefficient for SSOR preconditioner");
	}
	param.leave_subsection();
	param.enter_subsection("Postprocessing");
	{
		param.declare_entry("Apply postprocessing",
							"true",
							Patterns::Bool(),
							"Apply postprocessing?");
		param.declare_entry("Residual tolerance",
							"1e-10",
							Patterns::Double(),
							"Tolerance when checking for local and global conservation.");
		param.declare_entry("Update neumann boundary",
							"false",
							Patterns::Bool(),
							"Perform postprocessing on Neumann boundary?");
		param.declare_entry("Update dirichlet boundary",
							"true",
							Patterns::Bool(),
							"Perform postprocessing on Dirichlet boundary?");
		param.declare_entry("Harmonic weighting",
							"true",
							Patterns::Bool(),
							"Use harmonic weighting of flux on element edges?");
		param.declare_entry("Weighted L2-norm",
							"true",
							Patterns::Bool(),
							"Minimize correction in weighted L2-rnom?");
	}
	param.leave_subsection();
	param.enter_subsection("Transport solver");
	{
		param.declare_entry("Solver tolerance",
							"1e-10",
							Patterns::Double(),
							"Tolerance for linear solver");
		param.declare_entry("Do transport",
							"true",
							Patterns::Bool(),
							"Run transport solver?");
		param.declare_entry("Output file base",
							"solution_transport",
							Patterns::Anything(),
							"Where to store results from transport solver");
		param.declare_entry("Inflow concentration",
							"1.0",
							Patterns::Double(),
							"Inflow concentration on inflow boundary and from source");
		param.declare_entry("Use exact velocity",
							"false",
							Patterns::Bool(),
							"Use exact velocity in transport solver");
		param.declare_entry("Stride",
							"1",
							Patterns::Integer(),
							"Output every stride'th time step");
		param.declare_entry("Reference solution file",
							"",
							Patterns::FileName(),
							"File that stores reference solution (dealii binary mode). Blank ignores reference solution");
	}
	param.leave_subsection();
}

