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

#ifndef DIRICHLET_FLUX_RECOVERY_H
#define DIRICHLET_FLUX_RECOVERY_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <iostream>
#include <cmath>
#include <boost/concept_check.hpp>

#include <deal.II/base/logstream.h>

#include "HelpFunctions.h"

using namespace dealii;


// Routine for recovering flux on Dirichlet boundary
template <int dim>
class DirichletFluxRecovery
{
public:
	DirichletFluxRecovery(const DoFHandler<dim> &dh) : dof_handler(&dh) {}

	void set_parameters(unsigned int nq, unsigned int id = 0) { n_qpoints = nq; dirichlet_id = id; }

	void setup_and_assemble_matrix();

	template <class Matrix, class LAVector>
	void construct_rhs_steady(const Matrix* laplace_matrix, const LAVector* rhs_no_bcs, LAVector pressure_sol);

	template<class Matrix, class LAVector>
	void add_time_dependent_rhs(const Matrix* mass_matrix,
								const LAVector pressure, const LAVector pressure_old,
								double alpha, double dt);

	void solve();

	Vector<double> get_dirichlet_flux() { return dirichlet_flux_global; }
	void flux_values(const FEFaceValuesBase<dim>& fe_face_values, std::vector<double>& result);
	
	bool globally_conservative(const FE_Q<dim> fe, Function<dim>* neumann, Function<dim>* source, const double tol = 1e-5);
private:
	unsigned int dirichlet_id;
	
	unsigned int n_dirichlet_dofs;
	unsigned int n_qpoints;
	
	// Store solution
	const SmartPointer<const DoFHandler<dim>> dof_handler;
	Vector<double> dirichlet_flux_global;
	
	// Boundary map, set and mapping
	typename FunctionMap<dim>::type boundary_map;
	std::set<types::boundary_id> boundary_set;
	std::vector<types::global_dof_index> map_dof_to_boundary;
	std::vector<types::global_dof_index> dirichlet_dofs;
	
	// Linear system variables
	SparsityPattern sparsity_boundary;
	SparseMatrix<double> boundary_mass_matrix;
	Vector<double> boundary_rhs;
};


template <int dim>
void DirichletFluxRecovery<dim>::flux_values(const FEFaceValuesBase<dim>& fe_face_values, std::vector<double>& result)
{
	unsigned int n_q = fe_face_values.n_quadrature_points;
	AssertDimension(result.size(), n_q);	
	fe_face_values.get_function_values(dirichlet_flux_global, result);
}


template <int dim>
void DirichletFluxRecovery<dim>::setup_and_assemble_matrix()
{
	// Insert Dirichlet boundary to map and set. OBS! We set ZeroFunction since we don't actually use it later
	boundary_map[dirichlet_id] = new ZeroFunction<dim>();
	boundary_set.insert(dirichlet_id);
	
	n_dirichlet_dofs = dof_handler->n_boundary_dofs(boundary_map);
	
	// Create mapping from global dofs to boundary dofs
	DoFTools::map_dof_to_boundary_indices(*dof_handler, boundary_set, map_dof_to_boundary);
	
	// Sparisity
	DynamicSparsityPattern csp(n_dirichlet_dofs);
	DoFTools::make_boundary_sparsity_pattern(*dof_handler, boundary_map, map_dof_to_boundary, csp);
	sparsity_boundary.copy_from(csp);
	boundary_mass_matrix.reinit(sparsity_boundary);
	
	Vector<double> mass_rhs(n_dirichlet_dofs);
	MatrixCreator::create_boundary_mass_matrix(*dof_handler, QGauss<dim-1>(n_qpoints), boundary_mass_matrix, 
											   boundary_map, mass_rhs, map_dof_to_boundary);
	// Remove content of mass_rhs since we don't need it and since it is wrong (we specified ZeroFunction as Dirichlet function)
	mass_rhs.reinit(0);
	
	dirichlet_dofs.resize(n_dirichlet_dofs);
	IndexSet dirichlet_index_set(n_dirichlet_dofs);
	DoFTools::extract_boundary_dofs(*dof_handler, ComponentMask(), dirichlet_index_set, boundary_set);
	dirichlet_index_set.fill_index_vector(dirichlet_dofs);
}


template <int dim>
template <class Matrix, class LAVector>
void DirichletFluxRecovery<dim>::construct_rhs_steady(const Matrix* laplace_matrix, const LAVector* rhs_no_bcs, LAVector pressure_sol)
{
	AssertDimension(laplace_matrix->n(), pressure_sol.size());
	unsigned int n_global_dofs = pressure_sol.size();
	
	boundary_rhs.reinit(0);
	boundary_rhs.reinit(n_dirichlet_dofs);
	
	// Extract subvector from original rhs
	rhs_no_bcs->extract_subvector_to(dirichlet_dofs.begin(), dirichlet_dofs.end(), boundary_rhs.begin());
	
	// Extract local part of laplace_matrix * pressure_sol
	Vector<double> boundary_laplace_rhs(n_dirichlet_dofs);
	LAVector laplace_times_pressure;
	reinit_vec_seq(laplace_times_pressure, n_global_dofs);
	laplace_matrix->vmult(laplace_times_pressure, pressure_sol);
	laplace_times_pressure.extract_subvector_to(dirichlet_dofs.begin(), dirichlet_dofs.end(), boundary_laplace_rhs.begin());
	
	// Put together rhs
	boundary_rhs.add(-1.0, boundary_laplace_rhs);
}


// Add time dependent part to rhs:
// b = b + alpha/dt M(p_new - p_old)
template <int dim>
template <class Matrix, class LAVector>
void DirichletFluxRecovery<dim>::add_time_dependent_rhs(const Matrix* mass_matrix,
														const LAVector pressure, const LAVector pressure_old,
														double alpha, double dt)
{
	LAVector pressure_diff = pressure;
	pressure_diff -= pressure_old;
	LAVector mass_matrix_times_pressure_diff;
	reinit_vec_seq(mass_matrix_times_pressure_diff, pressure.size());
	Vector<double> boundary_mass_rhs(n_dirichlet_dofs);
	mass_matrix->vmult(mass_matrix_times_pressure_diff, pressure_diff);
	mass_matrix_times_pressure_diff.extract_subvector_to(dirichlet_dofs.begin(), dirichlet_dofs.end(), boundary_mass_rhs.begin());
	boundary_rhs.add(-alpha/dt, boundary_mass_rhs);
}



template <int dim>
void DirichletFluxRecovery<dim>::solve()
{
	Vector<double> dirichlet_flux_local(n_dirichlet_dofs);
	
	SolverControl      solver_control (n_dirichlet_dofs, 1e-15);
	SolverCG<>         solver (solver_control);
	PreconditionSSOR<> precondition;
	precondition.initialize(boundary_mass_matrix, 1.5);

	// Solve system
	solver.solve (boundary_mass_matrix, dirichlet_flux_local, boundary_rhs, precondition);
	
	// Distribute to global vector
	dirichlet_flux_global.reinit(dof_handler->n_dofs());
	dirichlet_flux_global.add(dirichlet_dofs, dirichlet_flux_local);
}


// TODO: This only works for steady problems
template <int dim>
bool DirichletFluxRecovery<dim>::globally_conservative(const FE_Q<dim> fe, Function<dim>* neumann, Function<dim>* source, const double tol)
{
	FEValues<dim>     fe_values(fe, QGauss<dim>(n_qpoints), update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> fe_face_values(fe, QGauss<dim-1>(n_qpoints), 
									 update_values | update_quadrature_points | update_JxW_values);
	
	const unsigned int n_qpoints_face = fe_face_values.n_quadrature_points;
	const unsigned int n_qpoints_cell = fe_values.n_quadrature_points;
	
	std::vector<double> face_flux(n_qpoints_face);
	
	double source_integrated = 0.0;
	double flux_out = 0.0;
	
	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler->begin_active(),
	end  = dof_handler->end();
	for ( ; cell!=end; ++cell) {
		fe_values.reinit(cell);
		for (unsigned int q=0; q<n_qpoints_cell; ++q) {
			source_integrated += source->value(fe_values.quadrature_point(q)) * fe_values.JxW(q);
		}
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if (cell->at_boundary(face)) {
				fe_face_values.reinit(cell, face);
				// Dirichlet boundary
				if (cell->face(face)->boundary_id() == dirichlet_id) {
					flux_values(fe_face_values, face_flux);
				}
				else {
					neumann->value_list(fe_face_values.get_quadrature_points(), face_flux);
				}
				for (unsigned int q=0; q<n_qpoints_face; ++q) {
					flux_out += face_flux[q] * fe_face_values.JxW(q);
				}
			}
		}
	}
	const double global_residual = source_integrated - flux_out;
	std::cout << "Global residual = Source - FluxOut = " << global_residual << std::endl;
	return (abs(global_residual) < tol);
}


#endif // DIRICHLET_FLUX_RECOVERY_H