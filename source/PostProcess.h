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

#ifndef POSTPROCESSBASE_H
#define POSTPROCESSBASE_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/timer.h>
#include <deal.II/numerics/vector_tools.h>

#include "InterfaceView.h"
#include "HelpFunctions.h"
#include "EmbeddedSurface.h"

#include <iostream>
#include <fstream>
#include <boost/concept_check.hpp>


using namespace dealii;

/*
 * Base class for postprocessing methods.
 * These methods take velocity fields that are not locally conservative
 * and add a correction to make them so.
 */
template <int dim>
class PostProcessBase
{
public:
	PostProcessBase(Triangulation<dim> &tria,
					const FE_FaceQ<dim> &fe_face,
					DoFHandler<dim> &dh_face);

	void set_parameters(ParameterHandler& prm);
	void set_fractures(double perm);

	virtual void setup_system() {}
	Vector<double> apply(Vector<double> residuals);

	virtual ~PostProcessBase();

	double get_residual_l2norm() const { return residual_l2norm; }
	double get_correction_l2_norm() const;
	double get_correction_edgenorm() const;
	
	virtual void print_timing() {}

protected:

	// Triangulation and piecewise constant FE space
	SmartPointer<Triangulation<dim>> triangulation;
	SmartPointer<const FE_FaceQ<dim>> fe_face;
	SmartPointer<DoFHandler<dim>> dh_face;

	// Actually store these here, since they are not used outside
	FE_DGQ<dim>     fe_cell;
	DoFHandler<dim> dh_cell;

	// Virtual solve method reimplemented in derived classes
	virtual void solve(Vector<double> /*residuals*/) {}

	// Common methods to all postprocessing classes
	void setup_constraints();
	void update_flux();
	void output_grid();

	void is_conservative(Vector<double> residuals);

	// Handle constraints
	// TODO: Needed anymore?
	ConstraintMatrix constraints;

	// Vector to store flux correction
	Vector<double> correction;

	SmartPointer<FractureNetwork> fracture_network;
	double perm_fracture;
	bool fractured = false;

	// Input parameters
	double residual_tol;
	bool update_neumann_bdry;
	bool pure_neumann;

	bool locally_conservative;
	double residual_l2norm;
};


/* Constructor takes a triangulation, finite element and dof handler.
 * Distributes dofs
 */
template <int dim>
PostProcessBase<dim>::PostProcessBase(Triangulation<dim> &tria,
									  const FE_FaceQ<dim> &fe_face,
									  DoFHandler<dim> &dh_face)
: triangulation(&tria),
  fe_face(&fe_face),
  dh_face(&dh_face),
  fe_cell(0),
  dh_cell(tria),
  fracture_network(new FractureNetwork())
{}


/* Destructor to clear dof handler for cell constants,
 * so that it does not point to triangulation any more
 */
template <int dim>
PostProcessBase<dim>::~PostProcessBase()
{
	dh_cell.clear();
}


// Set parameters from ParameterHandler
template <int dim>
void PostProcessBase<dim>::set_parameters(ParameterHandler& prm)
{
	prm.enter_subsection("Postprocessing");
	residual_tol = prm.get_double("Residual tolerance");
	update_neumann_bdry = prm.get_bool("Update neumann boundary");
	pure_neumann = !(prm.get_bool("Update dirichlet boundary"));
	prm.leave_subsection();
}


// Set fracture and fracture permeability
template <int dim>
void PostProcessBase<dim>::set_fractures(double perm)
{
	perm_fracture = perm;
	fractured = true;
}


// Calculate and return L2 norm of correction
template <int dim>
double PostProcessBase<dim>::get_correction_l2_norm() const
{
	return L2_norm_face(*triangulation, *dh_face, correction);
}


// Calculate and return edge norm of correction
template <int dim>
double PostProcessBase<dim>::get_correction_edgenorm() const
{
	return edgenorm(*triangulation, *dh_face, correction);
}


// User function. Apply postprocessing algorithm to input residuals
template <int dim>
Vector<double> PostProcessBase<dim>::apply(Vector<double> residuals)
{
	// Check that size of residuals is equal to number of cells
	Assert(residuals.size() == triangulation->n_active_cells(),
		   ExcDimensionMismatch(residuals.size(), triangulation->n_active_cells()));

	correction.reinit(dh_face->n_dofs());
	is_conservative(residuals);
	if (! locally_conservative) {
		solve(residuals);
		constraints.distribute(correction);
		std::cout << "Flux correction L2 norm:  " << L2_norm_face(*triangulation, *dh_face, correction) << std::endl;
		std::cout << "Flux correction edgenorm: " << edgenorm(*triangulation, *dh_face, correction) << std::endl;

		is_conservative(residuals);
		if (! locally_conservative)
			std::cout << "Warning: Flux correction is not locally conservative!" << std::endl;
	}
	else {
		std::cout << "Input residual is zero. Nothing to do." << std::endl;
	}
	return correction;;
}


/* Check if calculated correction gives conservative velocity.
 * Also, calculate residual L2 norm.
 */
template <int dim>
void PostProcessBase<dim>::is_conservative(Vector<double> residuals)
{
	locally_conservative = true;
	residual_l2norm = 0.0;

	// Loop through cells
	typename DoFHandler<dim>::active_cell_iterator
	cellc = dh_cell.begin_active(),
	cellf = dh_face->begin_active(),
	endc = dh_cell.end();
	for ( ; cellc != endc; ++cellc, ++cellf) {
		std::vector<unsigned int> face_dofs(GeometryInfo<dim>::faces_per_cell);
		cellf->get_dof_indices(face_dofs);
		std::vector<unsigned int> cell_dof(1);
		cellc->get_dof_indices(cell_dof);
		// Calculate new residual where the correction is used
		double new_residual = residuals(cell_dof[0]) * cellc->measure();
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			new_residual += GeometryInfo<dim>::unit_normal_orientation[face] * correction(face_dofs[face]) * face_measure<dim>(cellc, face);
		}
		residual_l2norm += pow(new_residual, 2.0) / cellc->measure();
		if (std::abs(new_residual) > residual_tol)
			locally_conservative = false;
	}
	residual_l2norm = sqrt(residual_l2norm);
}


// Output grid to grid.eps
template <int dim>
void PostProcessBase<dim>::output_grid()
{
  const std::string file = "grid.eps";
  std::ofstream out(file);
  GridOut grid_out;
  grid_out.write_eps(*triangulation, out);
}


// Setup constraints so that coarse face is equal to sum of finer faces
template <int dim>
void PostProcessBase<dim>::setup_constraints()
{
	constraints.clear();
	setup_flux_constraints_subfaces(*dh_face, constraints, false);
	constraints.close();
}


#endif // POSTPROCESSBASE_H
