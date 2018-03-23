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

#ifndef POSTPROCESSGS_H
#define POSTPROCESSGS_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_dgq.h>

#include "PostProcess.h"
#include "HelpFunctions.h"

#include <iostream>

using namespace dealii;

/* This is the Gauss-Seidel postprocessing approach based on
 *
 * S Sun and MF Wheeler (2005): "Projections of velocity data for the
 * 								 compatibility with transport"
 */
template <int dim>
class PostProcessGS : public PostProcessBase<dim>
{
public:

	typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;

	// Constructor calls base constructor and then initialize InterfaceView object
	PostProcessGS(Triangulation<dim> &tria,
				  const FE_FaceQ<dim> &fe_face,
				  DoFHandler<dim> &dh_face)
		: PostProcessBase<dim>(tria, fe_face, dh_face)
 		{
			ExcMessage("This class is deprecated.");
		}

	void set_parameters(double tol, bool update_neumann = false, bool update_dirichlet = true);

	void setup_system();
	
	void print_timing();

private:
	void solve(Vector<double> residuals);

	// Variable to access interface (intersection between neighboring cells) data
	InterfaceView<dim> interfaces;

	// alpha as defined on p. 666
	Vector<double> alpha;
	
	// Timer variables
	Timer t_setup;
	Timer t_iteration;
};


// Set parameters
template <int dim>
void PostProcessGS<dim>::set_parameters(double tol, bool update_neumann, bool update_dirichlet)
{
	this->residual_tol = tol;
	this->update_neumann_bdry = update_neumann;
	this->pure_neumann = !update_dirichlet;
}


// Initialize interfaces and calculate alpha's
template <int dim>
void PostProcessGS<dim>::setup_system()
{
	this->dh_cell.distribute_dofs(this->fe_cell);
	this->dh_face->distribute_dofs(*(this->fe_face));
	
	interfaces.initialize(*(this->triangulation), this->dh_cell, *(this->dh_face));
	this->setup_constraints();
	alpha.reinit(interfaces.n());
	for (unsigned int l=0; l<interfaces.n(); ++l) {
		const Interface<dim> interface = interfaces[l];
		const double cell1_meas = interface.get_cells().first->measure();
		const double cell2_meas = interface.get_cells().second->measure();
		alpha(l) = cell1_meas * cell2_meas / ( interface.get_measure() * (cell1_meas + cell2_meas) );
	}
}


/* Calculate sequences as defined by equations (4.4)-(4.9) in Sun and Wheeler (2006)
 * Takes cell-constant residuals as input
 */
template <int dim>
void PostProcessGS<dim>::solve(Vector<double> residuals)
{
	// TODO: Can this loops be made more efficient
	// TODO: Do we need the interface class?

	Vector<double> R_old = residuals;
	Vector<double> R(R_old);
	Vector<double> U_old(interfaces.n());
	Vector<double> U(U_old);

	unsigned int max_it = this->dh_cell.n_dofs() * 10;
	const double tol = this->residual_tol;
	bool converged = false;
	for (unsigned int k=0; k<max_it; ++k) {
		for (unsigned int l=0; l<interfaces.n(); ++l) {
			// Extract interface and some derived quantities
			const Interface<dim> interface = interfaces[l];
			const typename Interface<dim>::CellPair cells = interface.get_cells();
			std::vector<unsigned int> cell1_dof(this->fe_cell.dofs_per_cell);
			std::vector<unsigned int> cell2_dof(this->fe_cell.dofs_per_cell);
			cells.first->get_dof_indices(cell1_dof);
			cells.second->get_dof_indices(cell2_dof);
			// OBS! Need to be sure that interface normal points from cell1 to cell2.
			//      This is taken care of by InterfaceView
			double c_l = alpha[l] * ( R_old(cell2_dof[0]) - R_old(cell1_dof[0]) );

			// Update R
			R = R_old;
			R(cell1_dof[0]) += interface.get_measure() / cells.first->measure() * c_l;
			R(cell2_dof[0]) -= interface.get_measure() / cells.second->measure() * c_l;

			// Update U
			U = U_old;
			U(l) += c_l;

			// Update old
			U_old = U;
			R_old = R;
		}

		// Check if residual is smaller
		if (R.norm_sqr() > R_old.norm_sqr() ) {
			std::cout << "Warning: Residual norm did not decrease!" << std::endl;
		}
		Vector<double> update = U;
		update -= U_old;

		// Check if residual has converged to zero
		if ( (std::sqrt(R.norm_sqr()) < tol) && (std::sqrt(update.norm_sqr()) < tol) ) {
			std::cout << "GS algorithm converged in " << k << " steps." << std::endl;
			converged = true;
			break;
		}
	}

	if ( ! converged) {
		std::cout << "Warning: GS algorithm failed to converge in " << max_it << " steps." << std::endl;
	}

	// Add converged sequence to correction
	for (unsigned int l=0; l<interfaces.n(); ++l) {
		this->correction(interfaces.interface_to_face(l)) = U(l);
	}
}


// Print timing
template <int dim>
void PostProcessGS<dim>::print_timing()
{
	std::cout << "Timing postprocess GS (wall time in sec):" << std::endl
			  << "  Setup interfaces:     " << t_setup.wall_time() << std::endl
			  << "  Iterations:           " << t_iteration.wall_time() << std::endl
			  << "  Sum:                  " << t_setup.wall_time() + t_iteration.wall_time() << std::endl;	
}

#endif // POSTPROCESSGS_H

