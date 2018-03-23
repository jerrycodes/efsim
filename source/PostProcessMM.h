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

#ifndef POSTPROCESSMM_H
#define POSTPROCESSMM_H

#include <deal.II/base/config.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/base/timer.h>

#include "la_config.h"
#include "PostProcess.h"
#include "HelpFunctions.h"
#include "RockProperties.h"

#include <iostream>
#include <boost/concept_check.hpp>


using namespace dealii;


/* This is the minimum modification postprocessing approach based on
 *
 * S Sun and MF Wheeler (2005): "Projections of velocity data for the
 *								 compatibility with transport"
 */
template <int dim>
class PostProcessMM : public PostProcessBase<dim>
{
public:
	typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;

	// Constructor calls base constructor
	PostProcessMM(Triangulation<dim> &tria,
				  const FE_FaceQ<dim> &fe_face,
				  DoFHandler<dim> &dh_face,
				  const RockProperties<dim> &r)
		: PostProcessBase<dim>(tria, fe_face, dh_face)
		{
			rock = &r;
		}

	void set_parameters(ParameterHandler& prm);
	void set_parameters(double tol, bool update_neumann = false, bool update_dirichlet = true, bool weighted = true);

 	void setup_system();
	
	void print_timing();

private:
 	void solve(Vector<double> residuals);

 	SparsityPattern sparsity_pattern;
	LA::Matrix system_matrix;

 	// alpha_coeff represent the solution of the linear problem defined on p. 659
 	Vector<double> alpha_coeff;

	// Timer variables
	Timer t_assemble;
	Timer t_preconditioner;
	Timer t_solve;
	Timer t_apply;

	const RockProperties<dim>* rock;

	bool weighted_norm;

	double get_weight(CellIterator cell, CellIterator neighbor);
	double get_weight(CellIterator cell);
};


template <int dim>
void PostProcessMM<dim>::set_parameters(ParameterHandler& prm)
{
	PostProcessBase<dim>::set_parameters(prm);
	prm.enter_subsection("Postprocessing");
	weighted_norm = prm.get_bool("Weighted L2-norm");
	prm.leave_subsection();
}


template <int dim>
void PostProcessMM<dim>::set_parameters(double tol, bool update_neumann, bool update_dirichlet, bool weighted)
{
	this->residual_tol = tol;
	this->update_neumann_bdry = update_neumann;
	this->pure_neumann = !update_dirichlet;
	weighted_norm = weighted;
}


/* Setup system of linear equations, equation (3.4)
 * Notice that the global matrix is slightly different from Sun and Wheeler (2006)
 * due to a modified basis.
 * Diagonal elements are equal to circumference of element (potentially excluding Neumann boundary)
 * Off-diagonals are equal to -meas(face)
 */
template <int dim>
void PostProcessMM<dim>::setup_system()
{
	this->dh_cell.distribute_dofs(this->fe_cell);
	this->dh_face->distribute_dofs(*(this->fe_face));
	
	t_assemble.reset();
	t_assemble.start();
	
	this->setup_constraints();

	std::cout << "DoFs postprocessor: " << this->dh_cell.n_dofs() << std::endl;

	alpha_coeff.reinit (this->dh_cell.n_dofs());

	// Set sparsity pattern to be equal to face connections.
	DynamicSparsityPattern d_sparsity(this->dh_cell.n_dofs());
	GridTools::get_face_connectivity_of_cells<dim,dim>(*(this->triangulation), d_sparsity);
	sparsity_pattern.copy_from(d_sparsity);
	system_matrix.reinit (sparsity_pattern);
	
	CellIterator
	cell = this->dh_cell.begin_active(),
	endc = this->dh_cell.end();
	for (; cell!=endc; ++cell) {
		const unsigned int cell_dof = cell->dof_index(0);
		//double circumference = 0.0;
		double diag_element = 0.0;
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			double face_meas = face_measure<dim>(cell, face);
			if ( ! cell->at_boundary(face) ) {
				CellIterator neighbor;
				if (cell->neighbor(face)->active()) {
					neighbor = cell->neighbor(face);
					const unsigned int neighbor_dof = neighbor->dof_index(0);
					const double weight = get_weight(cell, neighbor);
					system_matrix.set(cell_dof, neighbor_dof, -face_meas * weight);
					diag_element += face_meas * weight;
				}
				else {
					for (unsigned int i=0; i<GeometryInfo<dim>::max_children_per_face; ++i) {
						neighbor = cell->neighbor_child_on_subface(face, i);
						const unsigned int neighbor_dof = neighbor->dof_index(0);
						face_meas = face_measure<dim>(neighbor, GeometryInfo<dim>::opposite_face[face]);
						const double weight = get_weight(cell, neighbor);
						system_matrix.set(cell_dof, neighbor_dof, -face_meas * weight);
						diag_element += face_meas * weight;
					}
				}
				//circumference += face_meas;
			}
			else if (this->pure_neumann)
				continue;
			else if (this->update_neumann_bdry || (cell->face(face)->boundary_id() != 1) ) {
				//circumference += face_meas;
				diag_element += face_meas * get_weight(cell);
			}
		}
		//system_matrix.set(cell_dof, cell_dof, circumference);
		system_matrix.set(cell_dof, cell_dof, diag_element);
	}
	
	t_assemble.stop();
	
	// If pure Neumann problem, set alpha(0) = 0
	// Remove all entries but the diagonal of first row and column (set rhs(0) = 0 in solve())
	if (this->pure_neumann) {
		// TODO: Is there a better way to do this?
		for (unsigned int i = 1; i < system_matrix.m(); ++i) {
			if (system_matrix.el(i,0) != 0.0)
				system_matrix.set(i, 0, 0.0);
			if (system_matrix.el(0,i) != 0.0)
				system_matrix.set(0, i, 0.0);
		}
	}
}


/* Solve linear problem and then calculate the correction from the basis
 * Note that we use a different basis than in Sun and Wheeler (2006)
 * Takes cell-constant residuals as input
 */
template <int dim>
void PostProcessMM<dim>::solve(Vector<double> residuals)
{
	// TODO: Examine different linear solvers
	// TODO: Check what linear solver tolerance need to be used to ensure residual tolerance
	// TODO: Only initialize solver and preconditioner once

	// Set up a PCG solver and solve system
	SolverControl solver_control (this->dh_cell.n_dofs(), this->residual_tol * 1e-2);
	LA::CG solver(solver_control);
	
	// Need to multiply residuals by size of element
	// TODO: Later, we should consider do this outside this class
	CellIterator
	cell = this->dh_cell.begin_active(),
	endc = this->dh_cell.end();
	for (; cell!=endc; ++cell) {
		const unsigned int cell_dof = cell->dof_index(0);
		residuals(cell_dof) *= cell->measure();	
	}
	
	// If pure Neumann problem, set residuals(0) = 0
	if (this->pure_neumann) {
		double residual_sum = std::accumulate(residuals.begin(), residuals.end(), 0.0);
		if (std::abs(residual_sum) > this->residual_tol) {
			std::cout << "WARNING! Sum of residuals vector is not zero as it should be for a pure Neumann problem" 
					  << " (sum = " << residual_sum << ")" << std::endl;
		}
		residuals(0) = 0.0;
	}
	
	t_preconditioner.reset(); 
	t_preconditioner.start();
#ifdef USE_TRILINOS
	LA::AMG precondition;
	precondition.initialize(system_matrix);
	std::cout << "Linear solver: CG-AMG (Trilinos)" << std::endl;
#else
	LA::SSOR precondition;
	const double relaxation = 1.5;
	precondition.initialize(system_matrix, relaxation);
	std::cout << "Linear solver: CG-SSOR (dealII)" << std::endl;
#endif
	t_preconditioner.stop();
	t_solve.reset();
	t_solve.start();
	solver.solve (system_matrix, alpha_coeff, residuals,
			      precondition);
	t_solve.stop();
	
	t_apply.reset();
	t_apply.start();

	CellIterator
	cellc = this->dh_cell.begin_active(),
	endcc = this->dh_cell.end(),
	cellf = this->dh_face->begin_active();
	for (; cellc!=endcc; ++cellc, ++cellf) {
		// cellc and cellf point to same cell, but different dofs
		std::vector<unsigned int> cell_dof(1);
		cellc->get_dof_indices(cell_dof);
		const double alpha = alpha_coeff(cell_dof[0]);
		std::vector<unsigned int> face_dofs(this->fe_face->dofs_per_cell);
		cellf->get_dof_indices(face_dofs);

		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			const int normal_orientation = GeometryInfo<dim>::unit_normal_orientation[face];
			// If at boundary
			if (cellc->at_boundary(face)) {
				/* Add contribution only if
				 * (a) Flag 'update_neumann_bdry' is true; or
				 * (b) We are not at Neumann boundary
				 */
				if (this->pure_neumann)
					continue;
				if (this->update_neumann_bdry || (cellc->face(face)->boundary_id() != 1) ) {
					const double weight = get_weight(cellc);
					this->correction(face_dofs[face]) += normal_orientation * alpha * (-1.0) * weight;
				}
			}
			// Else if neighbor is NOT finer (this face is handled "from the other side")
			else if (!cellc->face(face)->has_children()) {
				const double weight = get_weight(cellc, cellc->neighbor(face));
				this->correction(face_dofs[face]) += normal_orientation * alpha * (-1.0) * weight;
				if ( cellf->neighbor_is_coarser(face) ) {
					std::vector<unsigned int> neighbor_dof(1);
					cellc->neighbor(face)->get_dof_indices(neighbor_dof);
					const double alpha_neighbor = alpha_coeff(neighbor_dof[0]);
					this->correction(face_dofs[face]) -= normal_orientation * alpha_neighbor * (-1.0) * weight;
				}
			}
 		}
	}
	
	t_apply.stop();
}


template <int dim>
double PostProcessMM<dim>::get_weight(CellIterator cell, CellIterator neighbor)
{
	double weight = 1.0;
	if (weighted_norm) {
		// TODO: What about anisotropic permeability?
		double perm_cell, perm_neighbor;
		
		if (this->fractured && cell->material_id() == 1)
			perm_cell = this->perm_fracture;
		else
			perm_cell = rock->get_perm(cell)[0][0];
		
		if (this->fractured && neighbor->material_id() == 1)
			perm_neighbor = this->perm_fracture;
		else
			perm_neighbor = rock->get_perm(neighbor)[0][0];
		
		weight = 2.0 * perm_cell * perm_neighbor / (perm_cell + perm_neighbor);
	}
	return weight;
}


template <int dim>
double PostProcessMM<dim>::get_weight(CellIterator cell)
{
	double weight = 1.0;
	if (weighted_norm) {
		if ( this->fractured && (cell->material_id() == 1) )
			weight = this->perm_fracture;
		else
			weight = rock->get_perm(cell)[0][0];
	}
	return weight;
}



// Print timing
template <int dim>
void PostProcessMM<dim>::print_timing()
{
	std::cout << "Timing postprocess MM (wall time in sec):" << std::endl
			  << "  Assemble system:      " << t_assemble.wall_time() << std::endl
			  << "  Setup preconditioner: " << t_preconditioner.wall_time() << std::endl
			  << "  Solve system:         " << t_solve.wall_time() << std::endl
			  << "  Apply solution:       " << t_apply.wall_time() << std::endl
			  << "  Sum:                  " << t_assemble.wall_time() + t_preconditioner.wall_time() + t_solve.wall_time() + t_apply.wall_time() << std::endl;	
}

#endif // POSTPROCESSMM_H
