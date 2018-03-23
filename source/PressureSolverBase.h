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

#ifndef PRESSURE_SOLVER_BASE_H
#define PRESSURE_SOLVER_BASE_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <boost/concept_check.hpp>

#include "la_config.h"
#include "HelpFunctions.h"
#include "DirichletFluxRecovery.h"
#include "RockProperties.h"
#include "ProblemFunctions.h"
#include "Velocity.h"

#include <deal.II/base/logstream.h>


using namespace dealii;


/* Base class for pressure solver
 */
template <int dim>
class PressureSolverBase
{
public:
	PressureSolverBase(Triangulation<dim> &tria, const FE_Q<dim> &fe, DoFHandler<dim> &dh,
					   const RockProperties<dim> &r, ProblemFunctionsFlow<dim> &f);

	virtual ~PressureSolverBase() {};

	virtual void set_parameters(ParameterHandler& prm);
	virtual void set_parameters(double penalty, std::string filename, unsigned int nq, bool weak_boundary);
	virtual void set_pure_neumann() { pure_neumann = true; }
	virtual void set_fractures(FractureNetwork&, double, double) = 0;
	virtual void output_fracture_pressure() const = 0;
	
	virtual void setup_system();

	double pressure_l2_error();
	double gradp_l2_error();
	LA::Vec get_pressure_solution() const;

	void set_pressure_solution(LA::Vec pressure_in);
	
	virtual void solve_time_step(double dt) = 0;
	virtual void run() = 0;
	
	void project_solution() const;
	
	virtual void print_timing() = 0;

	const LA::Matrix* get_laplace_matrix() const { return &laplace_matrix; }
	const LA::Vec* get_rhs_no_bcs()     const { return &rhs_no_bcs; }

	void setup_dirichlet_flux_recovery();
	virtual Vector<double> apply_dirichlet_flux_recovery(double = 0);

	void calculate_velocity(VelocityData<dim>& velocity, const bool harmonic_weighting);
	virtual void add_fracture_velocity(VelocityData<dim>&, double) const = 0;

protected:
	// Assemble functions
	void assemble_laplace();
	void assemble_rhs();
	void apply_dirichlet_strongly(bool dual = false);
	
	virtual void create_sparsity_pattern();
	
	void solve_linsys(bool dual = false);
	
	void calculate_velocity(Vector<double> &velocity, const DoFHandler<dim> &dof_handler_velocity) const;
	void output_results() const;
	
	virtual std::string filename() const = 0;
	
	// Store the triangulation, finite element and dof handler as SmartPointers.
	// See Step 13 for further details.
	const SmartPointer<Triangulation<dim>> 	  triangulation;
	const SmartPointer<const FE_Q<dim>>		  fe;
	const SmartPointer<DoFHandler<dim>> dof_handler;
	
	const RockProperties<dim>* rock;
	ProblemFunctionsFlow<dim>* funs;
	
	// Linear system variables
	SparsityPattern      sparsity_pattern;
	LA::Matrix         system_matrix;
	LA::Matrix         laplace_matrix;
	ConstraintMatrix     constraints;
	LA::Vec              system_rhs;
	LA::Vec              rhs_no_bcs;
	
	// Solution vector
	LA::Vec pressure_sol;
	LA::Vec dual_sol;
	
	// Input parameters
	double 		 dirichlet_penalty;
	std::string  filename_base;
	unsigned int n_qpoints;
	double 	     s_form = 1.0;
	double		 ssor_relaxation = 1.5;
	double 		 linsol_tol = 1e-10;
	bool 		 weak_bcs;
	bool		 pure_neumann = false;
	
	DirichletFluxRecovery<dim> dirichlet_flux_recovery;
	bool use_dirichlet_flux = false;
	
	// Calculate flux on internal boundary (subfunction to calculate_velocity)
	void internal_flux(const FEFaceValuesBase<dim>& fe_face_values,
					   const FEFaceValuesBase<dim>& fe_face_values_neighbor,
					   const bool harmonic_weighting,
					   std::vector<double>& result) const;
};


/* The constructor takes as input references to a triangulation,
 * a finite element space, a dof_handler, and the problem type.
 */
template <int dim>
PressureSolverBase<dim>::PressureSolverBase(Triangulation<dim> &tria, const FE_Q<dim> &fe, DoFHandler<dim> &dh,
											const RockProperties<dim> &r, ProblemFunctionsFlow<dim> &f)
: triangulation(&tria),
  fe (&fe),
  dof_handler (&dh),
  rock(&r),
  funs(&f),
  dirichlet_flux_recovery(dh)
{}


// Read parameters from a ParameterHandler object
template <int dim>
void PressureSolverBase<dim>::set_parameters(ParameterHandler& prm)
{
	prm.enter_subsection("Pressure solver");
	dirichlet_penalty = prm.get_double("Dirichlet penalty");
	filename_base = prm.get("Output file base");
	s_form = prm.get_double("S form");
	weak_bcs = prm.get_bool("Weak BCs");
	linsol_tol = prm.get_double("Linear solver tolerance");
	ssor_relaxation = prm.get_double("SSOR relaxation coeff");
	prm.leave_subsection();
	prm.enter_subsection("Global");
	n_qpoints = prm.get_integer("No quadrature points");
	prm.leave_subsection();
}


// Read parameters
template <int dim>
void PressureSolverBase<dim>::set_parameters(double penalty, std::string filename, unsigned int nq, bool weak_boundary)
{
	dirichlet_penalty = penalty;
	filename_base = filename;
	n_qpoints = nq;
	weak_bcs = weak_boundary;
}


template <int dim>
void PressureSolverBase<dim>::set_pressure_solution(LA::Vec pressure_in)
{
	AssertDimension(pressure_in.size(), dof_handler->n_dofs());
	pressure_sol = pressure_in;
}



// Set up hanging nodes constraints and initialize linear system
template <int dim>
void PressureSolverBase<dim>::setup_system()
{
	dof_handler->distribute_dofs(*fe);
	
	// If strong BCs, set Dirichlet penalty to zero so that velocity (flux) is calculated correctly (without penalty term on Dirichlet boundary)
	if ( !weak_bcs )
		dirichlet_penalty = 0.0;

	// Make hanging dof constraints (Dirichlet are imposed weakly)
	constraints.clear();
	DoFTools::make_hanging_node_constraints(*dof_handler, constraints);
	constraints.close ();

	std::cout << "DoFs pressure: " << dof_handler->n_dofs() << std::endl;

	create_sparsity_pattern();

	system_matrix.reinit(sparsity_pattern);
	laplace_matrix.reinit(sparsity_pattern);

	reinit_vec_seq(pressure_sol, dof_handler->n_dofs());
	reinit_vec_seq(system_rhs,   dof_handler->n_dofs());
}


template <int dim>
void PressureSolverBase<dim>::create_sparsity_pattern()
{
	DynamicSparsityPattern c_sparsity(dof_handler->n_dofs());
	DoFTools::make_sparsity_pattern(*dof_handler, c_sparsity, constraints, false);
	sparsity_pattern.copy_from(c_sparsity);
}



// Solve linear system with PCG
template <int dim>
void PressureSolverBase<dim>::solve_linsys(bool dual)
{
	SolverControl solver_control (dof_handler->n_dofs(), linsol_tol);
	
	// Use Directsolver if strong Dirichlet and Trilinos solver since
	// it is not possible to guarantee symmetry in this case.
#ifdef USE_TRILINOS
	if (!weak_bcs) {
		TrilinosWrappers::SolverDirect solver(solver_control);
		
		Timer timer;
		timer.reset();
		timer.start();

		std::cout << "Linear solver: DirectSolver (Trilinos)" << std::endl;
		if (dual)
			solver.solve (system_matrix, dual_sol, system_rhs);
		else
			solver.solve (system_matrix, pressure_sol, system_rhs);
		timer.stop();
		
		std::cout << "Time linear solver: " << timer.wall_time() << std::endl;

		// Apply constraints
		if (dual)
			constraints.distribute(dual_sol);
		else
			constraints.distribute(pressure_sol);
		
		return;
	}
#endif
	
	// Set up solver and preconditioner
	LA::CG solver (solver_control);

	Timer timer;
	timer.reset();
	timer.start();

#ifdef USE_TRILINOS
	LA::AMG precondition;
	precondition.initialize(system_matrix);
	std::cout << "Linear solver: CG-AMG (Trilinos)" << std::endl;
#else
	LA::SSOR precondition;
	precondition.initialize(system_matrix, ssor_relaxation);
	std::cout << "Linear solver: CG-SSOR (dealII)" << std::endl;
#endif

	timer.stop();
	std::cout << "Time setup preconditioner: " << timer.wall_time() << std::endl;

	// Solve system
	timer.reset();
	timer.start();
	if (dual)
		solver.solve (system_matrix, dual_sol, system_rhs, precondition);
	else
		solver.solve (system_matrix, pressure_sol, system_rhs, precondition);
	timer.stop();

	std::cout << "Time linear solver: " << timer.wall_time() << std::endl;

	// Apply constraints
	if (dual)
		constraints.distribute(dual_sol);
	else
		constraints.distribute(pressure_sol);
}


template <int dim>
void PressureSolverBase<dim>::assemble_laplace()
{
	laplace_matrix = 0;

	QGauss<dim>   quadrature_formula(n_qpoints);
	QGauss<dim-1> face_quadrature_formula(n_qpoints);

	FEValues<dim> fe_values (*fe, quadrature_formula,
							 update_gradients |
							 update_JxW_values);
	
	FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
		  	   	  	  	  	  	  	  update_values |
									  update_gradients |
									  update_quadrature_points |
									  update_normal_vectors |
									  update_JxW_values);

	const unsigned int   dofs_per_cell = fe->dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();
	const unsigned int   n_q_face_points = face_quadrature_formula.size();

	FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	// Main loop
	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler->begin_active(),
	endc = dof_handler->end();
	for (; cell!=endc; ++cell) {
		fe_values.reinit (cell);
		cell_matrix = 0;
		Tensor<2,dim> permeability_cell = rock->get_perm(cell);

		// Cell integral: (K \nabla p, \nabla \phi)
		for (unsigned int i=0; i<dofs_per_cell; ++i) {
			for (unsigned int j=i; j<dofs_per_cell; ++j) {
				double add_data = 0.0;
				for (unsigned int q_point=0; q_point<n_q_points; ++q_point) {
					add_data += permeability_cell *
								fe_values.shape_grad (i, q_point) *
								fe_values.shape_grad (j, q_point) *
								fe_values.JxW (q_point);
				}
				cell_matrix(i,j) = add_data;
				cell_matrix(j,i) = add_data;
			}
		}

		// If weak BCs, add penalty terms to system_matrix (rhs is handled later)
		if (weak_bcs) {
			// Loop through cell faces
			for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no) {
				// If at Dirichlet boundary
				if (cell->at_boundary(face_no) && cell->face(face_no)->boundary_id()==0) {
					fe_face_values.reinit(cell,face_no);
					double face_meas = face_measure<dim>(cell, face_no);
					for (unsigned int i=0; i<dofs_per_cell; ++i) {
						for (unsigned int j=0; j<dofs_per_cell; ++j) {
							double add_data_1 = 0.0;
							double add_data_2 = 0.0;
							for (unsigned int q_point=0; q_point<n_q_face_points; ++q_point) {
								// J_{D,\sigma}(p,\phi)
								add_data_1 += fe->get_degree() * fe->get_degree() * (dirichlet_penalty/face_meas) *
											  fe_face_values.shape_value(i,q_point) *
											  fe_face_values.shape_value(j,q_point) *
											  fe_face_values.JxW (q_point);
								// (K \nabla p\cdot n, \phi)
								add_data_2 += (permeability_cell *
											   fe_face_values.shape_grad (j, q_point)) *
											   fe_face_values.normal_vector(q_point) *
											   fe_face_values.shape_value (i, q_point) *
											   fe_face_values.JxW (q_point);
							}
							cell_matrix(i,j) += add_data_1;
							cell_matrix(i,j) -= add_data_2;
							cell_matrix(j,i) -= s_form * add_data_2;
						}
					}
				}
			}
		}

		// Distribute local contributions to global system.
		// This function takes care of hanging nodes (and other constraints)
		cell->get_dof_indices (local_dof_indices);
		constraints.distribute_local_to_global(cell_matrix, local_dof_indices, laplace_matrix);
	}
}


// Assemble potentially time-dependent rhs including contributions from BCs
template <int dim>
void PressureSolverBase<dim>::assemble_rhs()
{
	QGauss<dim>   quadrature_formula(n_qpoints);
	QGauss<dim-1> face_quadrature_formula(n_qpoints);

	FEValues<dim> fe_values (*fe, quadrature_formula,
							 update_values |
							 update_quadrature_points |
							 update_JxW_values);

	FEFaceValues<dim> fe_face_values (*fe, face_quadrature_formula,
		  	   	  	  	  	  	  	  update_values |
									  update_gradients |
									  update_quadrature_points |
									  update_normal_vectors |
									  update_JxW_values);

	const unsigned int   dofs_per_cell = fe->dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();
	const unsigned int   n_q_face_points = face_quadrature_formula.size();

	Vector<double>       cell_rhs (dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	// Main loop
	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler->begin_active(),
	endc = dof_handler->end();
	for (; cell!=endc; ++cell) {
		cell_rhs = 0;
		fe_values.reinit(cell);
		const Tensor<2,dim> permeability_cell = rock->get_perm(cell);

		// Assemble rhs
		for (unsigned int q=0; q<n_q_points; ++q) {
			const double JxW = fe_values.JxW(q);
			const double rhs_val = funs->right_hand_side->value(fe_values.quadrature_point(q));
			for (unsigned int i = 0; i<dofs_per_cell; i++) {
				cell_rhs(i) += rhs_val * fe_values.shape_value(i,q) * JxW;
			}
		}

		// Loop through cell faces
		for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no) {
			// If at boundary
			if (cell->face(face_no)->at_boundary()) {
				fe_face_values.reinit(cell,face_no);
				// Neumann
				if (cell->face(face_no)->boundary_id() == 1) {
					for (unsigned int q_point = 0; q_point<n_q_face_points ; ++q_point) {
						const double neumann_val = funs->neumann->value(fe_face_values.quadrature_point(q_point));
						const double JxW = fe_face_values.JxW(q_point);
						for (unsigned int i = 0; i<dofs_per_cell; i++) {
							// [\phi, u_B]
							cell_rhs(i) -= neumann_val * fe_face_values.shape_value(i,q_point) * JxW;
						}
					}
				}
				// Dirichlet (only if weakly)
				else if (weak_bcs) {
					double face_meas = face_measure<dim>(cell, face_no);
					for (unsigned int q_point=0; q_point<n_q_face_points; ++q_point) {
						const double penalty_val = fe->get_degree() * fe->get_degree() * (dirichlet_penalty/face_meas);
						const double dirichlet_val = funs->dirichlet->value(fe_face_values.quadrature_point(q_point));
						const double JxW = fe_face_values.JxW (q_point);
						const Tensor<1,dim> normal_vec = fe_face_values.normal_vector(q_point);
						for (unsigned int i=0; i<dofs_per_cell; ++i) {
							// J_{D,\sigma}(p_B,\phi)
							cell_rhs(i) += penalty_val * fe_face_values.shape_value(i,q_point) * dirichlet_val * JxW;
							// s_form * (K \nabla p\cdot n, p_B)
							cell_rhs(i) -= s_form * (permeability_cell * fe_face_values.shape_grad (i, q_point)) *
										   normal_vec * dirichlet_val * JxW;
						}
					}
				}
			}
		}
		// Distribute local contributions to global system.
		// This function takes care of hanging nodes (and other constraints)
		cell->get_dof_indices (local_dof_indices);
		constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
	}
	rhs_no_bcs = system_rhs;
}


template <int dim>
void PressureSolverBase<dim>::apply_dirichlet_strongly(bool dual)
{
	bool elinimate_columns;
#ifdef USE_TRILINOS
	elinimate_columns = false;
#else
	elinimate_columns = true;
#endif
	
	std::map<types::global_dof_index,double> boundary_values;
	if (dual) {
		// Homogeneous Dirichlet conditions on whole boundary
		VectorTools::interpolate_boundary_values (*dof_handler, 0, ZeroFunction<dim>(), boundary_values);
		VectorTools::interpolate_boundary_values (*dof_handler, 1, ZeroFunction<dim>(), boundary_values);
		// TODO: This is slow for Trilinos matrices
		MatrixTools::apply_boundary_values (boundary_values,
											system_matrix,
											dual_sol,
											system_rhs,
											elinimate_columns);
	}
	else {
		VectorTools::interpolate_boundary_values (*dof_handler, 0, *(funs->dirichlet),  boundary_values);
		// TODO: This is slow for Trilinos matrices
		MatrixTools::apply_boundary_values (boundary_values,
											system_matrix,
											pressure_sol,
											system_rhs,
											elinimate_columns);
	}
}


// Calculate velocity approximation in the cell centers.
// TODO: Replace this and use member variable velocity
template <int dim>
void PressureSolverBase<dim>::calculate_velocity(Vector<double> &velocity, const DoFHandler<dim> &dof_handler_velocity) const
{
	QGauss<dim> quadrature_formula(1);
	FEValues<dim> fe_values (*fe, quadrature_formula,
			                 update_values   | update_gradients |
	                         update_quadrature_points);
	typename DoFHandler<dim>::active_cell_iterator
	cellp = dof_handler->begin_active(),
	cellv = dof_handler_velocity.begin_active(),
	endc  = dof_handler->end();
	for ( ; cellp!=endc; ++cellp, ++cellv) {
		fe_values.reinit(cellp);

		std::vector<Tensor<1,dim> > pressure_grad_cell(quadrature_formula.size());
		fe_values.get_function_gradients(pressure_sol, pressure_grad_cell);

		const Tensor<2,dim> cell_perm = rock->get_perm(cellp);

	    Assert(dim == dof_handler_velocity.get_fe().dofs_per_cell, ExcMessage("dofs per cell for velocity should be equal to dim.") );
	    Tensor<1,dim> velocity_t = - cell_perm * pressure_grad_cell[0];
	    Vector<double> velocity_cell(dim);
	    for (int d=0; d<dim; ++d)
	    	velocity_cell(d) = velocity_t[d];

	    cellv->distribute_local_to_global(velocity_cell, velocity);
	}
}


/* Output results to vtk file
 * - permeability
 * - porosity
 * - pressure
 * - error
 */
template <int dim>
void PressureSolverBase<dim>::output_results() const
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler (*dof_handler);

	// Store permeability in vector
	Vector<double> poro_vec  = copy_vector(rock->get_poro());
	Vector<double> permx_vec = copy_vector(rock->get_perm_comp(0));
	Vector<double> permy_vec, permz_vec;

	data_out.add_data_vector (poro_vec, "porosity");
	if (rock->is_isotropic()) {
		data_out.add_data_vector (permx_vec, "permeability_iso");
	}
	else {
		data_out.add_data_vector (permx_vec, "permeability_x");
		if (dim > 1) {
			permy_vec = copy_vector(rock->get_perm_comp(1));
			data_out.add_data_vector (permy_vec, "permeability_y");
		}
		if (dim > 2) {
			permz_vec = copy_vector(rock->get_perm_comp(2));
			data_out.add_data_vector (permz_vec, "permeability_z");
		}
	}

	data_out.add_data_vector (pressure_sol, "pressure");

	// Calculate error and its L2 norm
	Vector<double> error(dof_handler->n_dofs());
	if (dim == 2) {
		VectorTools::interpolate(*dof_handler, *(funs->exact_pressure), error);
		error -= pressure_sol;
		data_out.add_data_vector(error, "Error");
	}

	// Output velocity
	FESystem<dim> fe_velocity(FE_DGQ<dim>(0), dim);
	DoFHandler<dim> dof_handler_velocity(*triangulation);
	dof_handler_velocity.distribute_dofs(fe_velocity);
	Vector<double> velocity_sol(dof_handler_velocity.n_dofs());
	calculate_velocity(velocity_sol, dof_handler_velocity);

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
  	  	  component_type_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
	data_out.add_data_vector(dof_handler_velocity, velocity_sol, "Velocity", component_type_velocity);

	// Build patches and write to file
	data_out.build_patches ();
	std::ofstream output(filename().c_str());
	data_out.write_vtk(output);

	// Clear objects
	data_out.clear();
	dof_handler_velocity.clear();
}


template <int dim>
double PressureSolverBase<dim>::pressure_l2_error()
{
	QGauss<dim> quadrature(n_qpoints +1);
	Vector<double> error_per_cell(dof_handler->n_dofs());
	VectorTools::integrate_difference(*dof_handler, pressure_sol, *(funs->exact_pressure), error_per_cell, quadrature, VectorTools::NormType::L2_norm);
	const double l2_error = error_per_cell.l2_norm();

	return l2_error;
}


// Get-function to return pressure solution
template <int dim>
LA::Vec PressureSolverBase<dim>::get_pressure_solution() const
{
	return pressure_sol;
}


/* Calculate L2 error of gradp
 */
template <int dim>
double PressureSolverBase<dim>::gradp_l2_error()
{
	const UpdateFlags update_flags = update_values | update_quadrature_points | update_JxW_values | update_gradients;
	FEValues<dim> fe_values(*fe, QGauss<dim>(n_qpoints +1), update_flags);
	const unsigned int n_qpoints_cell = fe_values.n_quadrature_points;

	double gradp_error = 0.0;

	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler->begin_active(),
	endc = dof_handler->end();
	
	for ( ; cell!=endc; ++cell) {
		fe_values.reinit(cell);
		std::vector< Tensor<1,dim> > gradient_sol(n_qpoints_cell);
		fe_values.get_function_gradients(pressure_sol, gradient_sol);
		for (unsigned int q=0; q<n_qpoints_cell; ++q) {
			Tensor<1,dim> grap_diff_qpoint = gradient_sol[q] - funs->exact_gradient->value(fe_values.quadrature_point(q));
			gradp_error += grap_diff_qpoint.norm_square() * fe_values.JxW(q);
		}
	}
	return sqrt(gradp_error);
}


template <int dim>
void PressureSolverBase<dim>::setup_dirichlet_flux_recovery()
{
	dirichlet_flux_recovery.set_parameters(n_qpoints);
	dirichlet_flux_recovery.setup_and_assemble_matrix();
}


template <int dim>
Vector<double> PressureSolverBase<dim>::apply_dirichlet_flux_recovery(double)
{
	dirichlet_flux_recovery.construct_rhs_steady(&laplace_matrix, &rhs_no_bcs, pressure_sol);
	dirichlet_flux_recovery.solve();
	
	// Check that calculated flux is globally conservative
	Assert(dirichlet_flux_recovery.globally_conservative(*fe, funs->neumann, funs->right_hand_side),
		   ExcMessage("Dirichlet flux recovery did not produce globally conservative flux"));
	
	Vector<double> dirichlet_flux = dirichlet_flux_recovery.get_dirichlet_flux();
	use_dirichlet_flux = true;
	return dirichlet_flux;
}


// Calculate velocity (RT0) from pressure solution with optional harmonic weighting
template <int dim>
void PressureSolverBase<dim>::calculate_velocity(VelocityData<dim>& velocity, const bool harmonic_weighting)
{
	const UpdateFlags update_flags_face    = update_values |
											 update_gradients |
											 update_quadrature_points |
											 update_normal_vectors |
											 update_JxW_values;
	const UpdateFlags update_flags_subface = update_gradients |
											 update_quadrature_points |
											 update_normal_vectors |
											 update_JxW_values;
	const UpdateFlags update_flags_face_neighbor = update_gradients;
	
	// Construct FEValues objects to access solution and mapping
	FEFaceValues<dim>    fe_face_values(*fe, QGauss<dim-1>(n_qpoints), update_flags_face);
	FEFaceValues<dim>    fe_face_values_neighbor(*fe, QGauss<dim-1>(n_qpoints), update_flags_face_neighbor);
	FESubfaceValues<dim> fe_subface_values(*fe, QGauss<dim-1>(n_qpoints), update_flags_subface);
	
	typename VelocityData<dim>::FE_Pointer fe_velocity = velocity.get_fe();
	typename VelocityData<dim>::DH_Pointer dh_velocity = velocity.get_dh();
	
	Assert(fe_velocity->degree == 1, ExcMessage("Only implemented for lowest degree RT"));
	
	std::vector<unsigned int> face_dofs(fe_velocity->n_dofs_per_cell());
	const unsigned int n_qpoints_face = fe_face_values.n_quadrature_points;
	
	typename DoFHandler<dim>::active_cell_iterator
	cellp = dof_handler->begin_active(),
	cellv = dh_velocity->begin_active(),
	endc  = dof_handler->end();
	for ( ; cellp!=endc; ++cellp, ++cellv) {
		cellv->get_dof_indices(face_dofs);
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			fe_face_values.reinit(cellp, face);
			std::vector<double> u_dot_n(n_qpoints_face);
			double face_flux = 0.0;
			
			// See step 30 for more documentation of looping process
			if (cellp->at_boundary(face)) {
				// (a) Neumann boundary
				if (cellp->face(face)->boundary_id() == 1) {
					for (unsigned int q=0; q<n_qpoints_face; ++q) {
						u_dot_n[q] = funs->neumann->value(fe_face_values.quadrature_point(q));
					}
				}
				// (b) Dirichlet boundary
				else {
					if ( use_dirichlet_flux ) {
						fe_face_values.get_function_values(dirichlet_flux_recovery.get_dirichlet_flux(), u_dot_n);
					}
					else {
						// Get pressure and gradient values
						std::vector<double> pressure_values(n_qpoints_face);
						std::vector<Tensor<1,dim>> pressure_gradients(n_qpoints_face);
						if ( weak_bcs ) fe_face_values.get_function_values(pressure_sol, pressure_values);
						fe_face_values.get_function_gradients(pressure_sol, pressure_gradients);
						
						const Tensor<1,dim> unit_normal = fe_face_values.normal_vector(0);
						const Tensor<2,dim> permeability_cell = rock->get_perm(fe_face_values.get_cell());
						
						// Calculate - K \nabla p \cdot n + dirichlet_penalty * (p - p_B)
						for (unsigned int q=0; q<n_qpoints_face; ++q) {
							u_dot_n[q] = - permeability_cell * pressure_gradients[q] * unit_normal;
							if (weak_bcs) {
								const Point<dim> qpoint = fe_face_values.quadrature_point(q);
								const double penalty_coeff_face = dirichlet_penalty / face_measure<dim>(fe_face_values.get_cell(), face);
								u_dot_n[q] += penalty_coeff_face * (pressure_values[q] - funs->dirichlet->value(qpoint));
							}
						}
					}
				}
				for (unsigned int q=0; q<n_qpoints_face; ++q) {
					face_flux += u_dot_n[q] * fe_face_values.JxW(q);
				}
			}
			else {
				// (c) If neighbors are finer: we need to loop through all subfaces
				if ( cellp->face(face)->has_children() ) {
					unsigned int neighbor_face = cellp->neighbor_face_no(face);
					for (unsigned int subface=0; subface<cellp->face(face)->number_of_children(); ++subface) {
						fe_subface_values.reinit(cellp, face, subface);
						typename DoFHandler<dim>::active_cell_iterator neighbor = cellp->neighbor_child_on_subface(face, subface);
						fe_face_values_neighbor.reinit(neighbor, neighbor_face);
						
						internal_flux(fe_subface_values, fe_face_values_neighbor, harmonic_weighting, u_dot_n);
						for (unsigned int q=0; q<n_qpoints; ++q) {
							face_flux += u_dot_n[q] * fe_subface_values.JxW(q);
						}
					}
				}
				else {
					// (d) If neighbor is coarser
					if ( cellp->neighbor_is_coarser(face) ) {
						const std::pair<unsigned int, unsigned int> face_info = cellp->neighbor_of_coarser_neighbor(face);
						const unsigned int neighbor_face = face_info.first;
						const unsigned int neighbor_subface = face_info.second;
						fe_subface_values.reinit(cellp->neighbor(face), neighbor_face, neighbor_subface);
						internal_flux(fe_face_values, fe_subface_values, harmonic_weighting, u_dot_n);
					}
					// (e) If neighbors are equally coarse
					else {
						unsigned int neighbor_face = cellp->neighbor_face_no(face);
						fe_face_values_neighbor.reinit(cellp->neighbor(face), neighbor_face);
						internal_flux(fe_face_values, fe_face_values_neighbor, harmonic_weighting, u_dot_n);
					}
					for (unsigned int q=0; q<n_qpoints_face; ++q) {
						face_flux += u_dot_n[q] * fe_face_values.JxW(q);
					}
				}
			}
			velocity.set_dof_value(face_dofs[face], GeometryInfo<dim>::unit_normal_orientation[face] * face_flux);
		}
	}
	velocity.apply_constraints();
}


/* Calculate u * n on a internal face.
 * Inputed FEFaceValuesBase objects should be reinit'ed to a the two faces on the common interface
 * Return results at all quadrature points in vector 'result'.
 */
template <int dim>
void PressureSolverBase<dim>::internal_flux(const FEFaceValuesBase<dim>& fe_face_values,
											const FEFaceValuesBase<dim>& fe_face_values_neighbor,
											const bool harmonic_weighting,
											std::vector<double>& result) const
{
	const unsigned int n_qpoints_face = fe_face_values.n_quadrature_points;
	
	// Get pressure and gradient values
	std::vector<Tensor<1,dim>> pressure_gradients(n_qpoints_face);
	std::vector<Tensor<1,dim>> pressure_gradients_neighbor(n_qpoints_face);
	fe_face_values.get_function_gradients(pressure_sol, pressure_gradients);
	fe_face_values_neighbor.get_function_gradients(pressure_sol, pressure_gradients_neighbor);
	
	const Tensor<1,dim> unit_normal = fe_face_values.normal_vector(0);
	const Tensor<2,dim> permeability_cell     = rock->get_perm(fe_face_values.get_cell());
	const Tensor<2,dim> permeability_neighbor = rock->get_perm(fe_face_values_neighbor.get_cell());
	
	double w1 = 0.5;
	//TODO: Only works for isotropic permeability
	if (harmonic_weighting) {
		w1 = permeability_neighbor[0][0] / ( permeability_cell[0][0] + permeability_neighbor[0][0] );
	}
	double w2 = 1.0 - w1;
	
	// Calculate {- K \nabla p \cdot n }
	for (unsigned int q=0; q<n_qpoints_face; ++q) {
		const double u_dot_n_1 = -permeability_cell * pressure_gradients[q] * unit_normal;
		const double u_dot_n_2 = -permeability_neighbor * pressure_gradients_neighbor[q] * unit_normal;
		result[q] = w1 * u_dot_n_1 + w2 * u_dot_n_2;
	}
}


template <int dim>
void PressureSolverBase<dim>::project_solution() const
{
	Triangulation<dim> tria_refined;
	tria_refined.copy_triangulation(*triangulation);
	DoFHandler<dim> dh_refined(tria_refined);
	dh_refined.distribute_dofs(*fe);
	Vector<double> solution_ref;
	Vector<double> solution_prev(pressure_sol);
	
	// Refine coarse elements first
	if (tria_refined.has_hanging_nodes()) {
		unsigned int max_level = tria_refined.n_global_levels();
		typename Triangulation<dim>::active_cell_iterator cell, endc;
		for (unsigned int l=0; l<max_level-1; ++l) {
			bool active_level = false;
			for (cell=tria_refined.begin_active(l); cell!=tria_refined.end_active(l); ++cell) {
				cell->set_refine_flag();
				active_level = true;
			}
			if (active_level) {
				SolutionTransfer<dim> solution_transfer(dh_refined);
				tria_refined.prepare_coarsening_and_refinement();
				solution_transfer.prepare_for_pure_refinement();
				tria_refined.execute_coarsening_and_refinement();
				dh_refined.distribute_dofs(*fe);
				solution_ref.reinit(dh_refined.n_dofs());
				solution_transfer.refine_interpolate(solution_prev, solution_ref);
				solution_prev = solution_ref;
			}
		}
	}
	
	// Transer solution to finer mesh (globally refined ref_cycles times)
	const unsigned int ref_cycles = 3;
	for (unsigned int cycle=0; cycle<ref_cycles; ++cycle) {
		SolutionTransfer<dim> solution_transfer(dh_refined);
		tria_refined.set_all_refine_flags();
		tria_refined.prepare_coarsening_and_refinement();
		solution_transfer.prepare_for_pure_refinement();
		tria_refined.execute_coarsening_and_refinement();
		dh_refined.distribute_dofs(*fe);
		solution_ref.reinit(dh_refined.n_dofs());
		solution_transfer.refine_interpolate(solution_prev, solution_ref);
		solution_prev = solution_ref;
	}
	
	// Output solution
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dh_refined);
	data_out.add_data_vector(solution_ref, "pressure");
	data_out.build_patches ();
	
	std::string file = filename();
	file.insert(file.rfind("-"), "_interpolated");
	std::ofstream output(file.c_str());
	
	data_out.write_vtk(output);

	data_out.clear();
	dh_refined.clear();
}


#endif // PRESSURE_SOLVER_BASE_H