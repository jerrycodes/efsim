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

#ifndef PARABOLIC_PRESSURE_SOLVER_H
#define PARABOLIC_PRESSURE_SOLVER_H


#include "PressureSolverBase.h"


using namespace dealii;


// Solver for parabolic (dynamic) pressure equation
template <int dim>
class ParabolicPressureSolver : public PressureSolverBase<dim>
{
public:
	// Constructor calls base constructor
	ParabolicPressureSolver(Triangulation<dim> &tria, const FE_Q<dim> &fe, DoFHandler<dim> &dh,
							const RockProperties<dim> &r, ProblemFunctionsFlow<dim> &f)
		: PressureSolverBase<dim>(tria, fe, dh, r, f)
		{}

	~ParabolicPressureSolver() {}

	void set_parameters(double penalty, std::string filename, unsigned int nq, bool weak_boundary);
	void set_parameters(ParameterHandler& prm);
	void set_pure_neumann() { Assert(false, ExcMessage("Pure Neumann problem not implemented for Parabolic solver yet")); }
	void set_fractures(FractureNetwork&, double, double) { ExcNotImplemented(); }
	void output_fracture_pressure() const { ExcNotImplemented(); }
	void add_fracture_velocity(VelocityData<dim>&, double) const { ExcNotImplemented(); }
	
	void setup_system();
	void initialize();
	void solve_time_step(double dt);

	void run(double dt, double end_time);

	// Dummy function that only throws an exception. This is to comply with EllipticPressureSolver 
	void run() { ExcInternalError(); }

	void print_timing();
	
	Vector<double> apply_dirichlet_flux_recovery(double dt);

private:
	void assemble_mass_matrix();
	
	void increment_time(double dt);
	
	std::string filename() const;
	
	// Store old solution
	LA::Vec pressure_sol_old;
	
	// Time variables
	double time;
	unsigned int time_step_count;
	
	// Additional linear system variables
	LA::Matrix mass_matrix;
	
	// Input parameters
	double alpha = 1.0;
	
	// Timer variables
	Timer t_assemble_static;
	Timer t_assemble_timedep;
	Timer t_solve;
};


// Read parameters 
template <int dim>
void ParabolicPressureSolver<dim>::set_parameters(double penalty, std::string filename, unsigned int nq, bool weak_boundary)
{
	PressureSolverBase<dim>::set_parameters(penalty, filename, nq, weak_boundary);
}


// Read parameters from a ParameterHandler object
template <int dim>
void ParabolicPressureSolver<dim>::set_parameters(ParameterHandler& prm)
{
	PressureSolverBase<dim>::set_parameters(prm);
	prm.enter_subsection("Pressure solver");
	alpha = prm.get_double("Alpha");
	prm.leave_subsection();
}


// Set up hanging nodes constraints and initialize linear system
template <int dim>
void ParabolicPressureSolver<dim>::setup_system()
{
	PressureSolverBase<dim>::setup_system();
	mass_matrix.reinit(this->sparsity_pattern);
	initialize();
}


// Initialize system to t = 0;
template <int dim>
void ParabolicPressureSolver<dim>::initialize()
{
	// Set solution vectors to initial values
	VectorTools::interpolate(*(this->dof_handler), *(this->funs->initial_pressure), this->pressure_sol);

	time = 0.0;
	time_step_count = 0;

	this->output_results();
	
	t_assemble_static.start();
	
	this->assemble_laplace();
	assemble_mass_matrix();
	
	t_assemble_static.stop();
}


// Perform one time step of pressure solver
template <int dim>
void ParabolicPressureSolver<dim>::solve_time_step(double dt)
{
	increment_time(dt);

	pressure_sol_old = this->pressure_sol;

	std::cout << "Time step " << time_step_count << " (t = " << time << "s)" << std::endl;
	
	// system_matrix = dt*laplace_matrix + alpha*mass_matrix;
	this->system_matrix = 0;
	this->system_matrix.add(dt, this->laplace_matrix);
	this->system_matrix.add(alpha, mass_matrix);

	t_assemble_timedep.reset();
	t_assemble_timedep.start();
	
	// Put rhs vector together
	this->system_rhs = 0;
	this->assemble_rhs();
	this->system_rhs *= dt;
	LA::Vec pressure_sol_old_times_alpha = pressure_sol_old;
	pressure_sol_old_times_alpha *= alpha;
	mass_matrix.vmult_add(this->system_rhs, pressure_sol_old_times_alpha);

	if (! this->weak_bcs) this->apply_dirichlet_strongly();

	t_assemble_timedep.stop();
	
	// Then solve matrix system
	t_solve.reset();
	t_solve.start();
	this->solve_linsys();
	t_solve.stop();

	// Output results
	this->output_results();

	std::cout << "  Pressure L2 error: " << this->pressure_l2_error() << std::endl;
}


// Assemble mass matrix. Use DealII functionality
template <int dim>
void ParabolicPressureSolver<dim>::assemble_mass_matrix()
{
	mass_matrix = 0;

	// MatrixCreator is not compatible with Trilinos SparseMatrix.
	// TODO: MatrixCreator is a little faster (factor ~1.23). Why?
#ifndef USE_TRILINOS
	MatrixCreator::create_mass_matrix(*(this->dof_handler),
	                                  QGauss<dim>(this->n_qpoints),
									  mass_matrix,
									  (const Function<dim>*)0,
									  this->constraints);
#else
	QGauss<dim> quadrature(this->n_qpoints);
	FEValues<dim> fe_values (*(this->fe), quadrature,
							 update_values | update_JxW_values);

	const unsigned int   dofs_per_cell = this->fe->dofs_per_cell;
	const unsigned int   n_q_points    = quadrature.size();
	FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	// Main loop
	typename DoFHandler<dim>::active_cell_iterator
	cell = this->dof_handler->begin_active(),
	endc = this->dof_handler->end();
	for (; cell!=endc; ++cell) {
		fe_values.reinit(cell);
		cell_matrix = 0;
		// Cell integral: (p, \phi)
		// Symmetric operation
		for (unsigned int i=0; i<dofs_per_cell; ++i) {
			for (unsigned int j=i; j<dofs_per_cell; ++j) {
				double add_data = 0.0;
				for (unsigned int q_point=0; q_point<n_q_points; ++q_point) {
					add_data += fe_values.shape_value(i, q_point) *
								fe_values.shape_value(j, q_point) *
								fe_values.JxW(q_point);
				}
				cell_matrix(i,j) = add_data;
				cell_matrix(j,i) = add_data;
			}
		}
		cell->get_dof_indices(local_dof_indices);
		this->constraints.distribute_local_to_global(cell_matrix, local_dof_indices, mass_matrix);
	}
#endif
}


template <int dim>
void ParabolicPressureSolver<dim>::increment_time(double dt)
{
	time += dt;
	++time_step_count;
	this->funs->set_time(time);
}


template <int dim>
std::string ParabolicPressureSolver<dim>::filename() const 
{
	std::ostringstream name;
	name << "output/" << this->filename_base << "-" << time_step_count << ".vtk";
	return name.str();
}


// Run solver from t=0 to t=end_time with constant time_steps dt
template <int dim>
void ParabolicPressureSolver<dim>::run(double dt, double end_time)
{
	setup_system();

	while (time < (end_time-dt + dt/100)) {
		solve_time_step(dt);
	}

	this->output_results();
}


// Print timing 
template <int dim>
void ParabolicPressureSolver<dim>::print_timing()
{
	std::cout << "Timing parabolic pressure solver (wall time in sec):" << std::endl
			  << "  Assemble static:    " << t_assemble_static.wall_time() << std::endl
			  << "  Assemble dynamic:   " << t_assemble_timedep.wall_time() << std::endl
			  << "  Solve system:       " << t_solve.wall_time() << std::endl
			  << "  Sum:                " << t_assemble_static.wall_time() + t_assemble_timedep.wall_time() + t_solve.wall_time() << std::endl;
}


template <int dim>
Vector<double> ParabolicPressureSolver<dim>::apply_dirichlet_flux_recovery(double dt)
{
	this->dirichlet_flux_recovery.construct_rhs_steady(&(this->laplace_matrix), &(this->rhs_no_bcs), this->pressure_sol);
	this->dirichlet_flux_recovery.add_time_dependent_rhs(&mass_matrix, this->pressure_sol, pressure_sol_old, alpha, dt);
	this->dirichlet_flux_recovery.solve();
	
	Vector<double> dirichlet_flux = this->dirichlet_flux_recovery.get_dirichlet_flux();
	this->use_dirichlet_flux = true;
	return dirichlet_flux;
}



#endif // PARABOLIC_PRESSURE_SOLVER_H

