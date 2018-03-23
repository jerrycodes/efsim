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

#ifndef INTERFACEVIEW_H
#define INTERFACEVIEW_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_dgq.h>

#include "HelpFunctions.h"

#include <iostream>

using namespace dealii;

/* Class to store interface data
 *
 * Need this class since dealII don't care about interfaces (only faces
 * associated with a cell)
 * We are particularly interested in the neighboring cells for each interface
 */
template<int dim>
class Interface
{
public:
	typedef typename DoFHandler<dim>::active_cell_iterator CellIterator;
	typedef typename std::pair<CellIterator, CellIterator> CellPair;

	Interface(double meas, CellIterator cell1, CellIterator cell2, unsigned int dof)
	: measure(meas),
	  cells(cell1, cell2),
	  global_dof_index(dof)
	{}

	// Get functions
	double get_measure() const { return measure; }
	CellPair get_cells() const { return cells; }

	unsigned int get_dof_index() const { return global_dof_index; }

private:
	double measure;
	CellPair cells; // Unit normal always point from cell1 to cell2
	unsigned int global_dof_index;
};

/*
 * Class to access interface data
 *
 * An interface is defined as the intersection between two neighboring cells.
 * We use DoFHandler instead of Triangulation to initialize this class since
 * we want access to cell dofs. But we need Triangulation to call clear_user_flags()
 * since DoFHandler::get_tria() is const.
*/
template <int dim>
class InterfaceView
{
public:
	InterfaceView()
	{}

	void initialize(Triangulation<dim>& triangulation, const DoFHandler<dim>& dh_cell, const DoFHandler<dim>& dh_face);

	// Overload [] operator for acces to face l
	Interface<dim> operator[](int l) { return interfaces[l]; }

	// Return number of interfaces
	unsigned int n() { return n_interfaces; }

	// Mapping from interface to face dof
	unsigned int interface_to_face(int l) const { return interfaces[l].get_dof_index(); }

private:
	unsigned int n_interfaces;
	std::vector<Interface<dim>> interfaces;
};


/* Loop over all cells and its neighbor to identify and initialize all interfaces
 * Note that each interface (cell intersection) is only stored once (no duplicates)
 */
template <int dim>
void InterfaceView<dim>::initialize(Triangulation<dim>& triangulation, const DoFHandler<dim>& dh_cell, const DoFHandler<dim>& dh_face)
{
	triangulation.clear_user_flags();
	n_interfaces = 0;

	typename Interface<dim>::CellIterator
	cellc = dh_cell.begin_active(),
	cellf = dh_face.begin_active(),
	endc  = dh_cell.end();

	for ( ; cellc != endc; ++cellc, ++cellf) {
		std::vector<unsigned int> face_dofs(dh_face.get_fe().dofs_per_cell);
		cellf->get_dof_indices(face_dofs);
		// Extract cell data
		for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
			// If neighbor at boundary or neighbor is not active (i.e. is refined)
			if ( cellc->at_boundary(face) )
				continue;
			// Is neighbor not active, skip this face, since neighbor has children
			// Instead visit this interface from the finer cell later
			if ( ! cellc->neighbor(face)->active() ) {
				continue;
			}
			const typename Interface<dim>::CellIterator neighbor = cellc->neighbor(face);
			// If neighbor not visited or neighbor is coarser
			if ( ( ! neighbor->user_flag_set() ) || ( cellc->neighbor_is_coarser(face) ) ) {
				const double measure = face_measure<dim>(cellc, face);
				if (GeometryInfo<dim>::unit_normal_orientation[face] == 1)
					interfaces.push_back(Interface<dim>(measure, cellc, neighbor, face_dofs[face]));
				else
					interfaces.push_back(Interface<dim>(measure, neighbor, cellc, face_dofs[face]));

				++n_interfaces;
				cellc->set_user_flag();
			}
		}
	}
	triangulation.clear_user_flags();
}


#endif // INTERFACEVIEW_H
