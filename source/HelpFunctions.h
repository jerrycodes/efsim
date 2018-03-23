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

#ifndef HELP_FUNCTIONS_H
#define HELP_FUNCTIONS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/base/index_set.h>

#include "la_config.h"
#include "ProblemFunctions.h"
#include "EmbeddedSurface.h"


using namespace dealii;


// Utility functions for grid construction, refinement, norms, boundary conditions, etc.


/* Calculate face measure of face on given cell.
 * Surprisingly, Deal II does not provide functionality to calculate face measure in 3D.
 */
template<int dim, class CellPointer>
double face_measure(CellPointer cell, unsigned int face)
{
	double face_measure;
	// If dim < 3, use Deal II function
	if (dim == 3) {
		// Extract vertices of face.
		std::vector<Point<dim>> vertices(4);
		for (int v=0; v<4; ++v) {
			vertices[v] = cell->face(face)->vertex(v);
		}
		const Tensor<1,dim> diag1 = Tensor<1,dim>(vertices[3]) - Tensor<1,dim>(vertices[0]);
		const Tensor<1,dim> diag2 = Tensor<1,dim>(vertices[2]) - Tensor<1,dim>(vertices[1]);
		Tensor<1,dim> cross_p = cross_product_3d(diag1, diag2);
		// |face| = 0.5 * || diag1 x diag2 ||
		face_measure = 0.5 * cross_p.norm();
	}
	else {
		face_measure = cell->face(face)->measure();
	}
	return face_measure;
}


/* Create triangulation by refining a square and optionally
 * refine locally and distort grid.
 * Also, define boundary faces as either Neumann or Dirichlet.
 */
// TODO: Get rid of ProblemType argument
template <int dim>
void make_grid (Triangulation<dim>& triangulation,
				ProblemType,
		        int global_refinement = 4,
				bool refine_locally = false,
				double distort_factor = 0.0)
{
	// Create cube and refine global
	GridGenerator::hyper_cube (triangulation, 0, 1);
	triangulation.refine_global (global_refinement);

	// Refine locally if wanted
	if (refine_locally) {
		typename Triangulation<dim>::active_cell_iterator
		cell = triangulation.begin_active(),
		endc = triangulation.end();
		for ( ; cell!=endc ; cell++) {
			/*
			if ( ( (cell->center()(0)>1.0/4.0) && (cell->center()(0)<3.0/4.0) ) &&
			     ( (cell->center()(1)>1.0/8.0) && (cell->center()(1)<7.0/8.0) ) )
				cell->set_refine_flag();
			if ( ( (cell->center()(0)>7.0/16.0) && (cell->center()(0)<9.0/16.0) ) &&
			     ( (cell->center()(1)>5.0/16.0) && (cell->center()(1)<11.0/16.0) ) )
				cell->clear_refine_flag();
			*/
			if ( ((cell->center()(0) < 0.5) && (cell->center()(1) < 0.5)) ||
				 ((cell->center()(0) > 0.5) && (cell->center()(1) > 0.5)) )
				cell->set_refine_flag();
		}
		triangulation.execute_coarsening_and_refinement();
	}

	// Make distorted grid
	const double diameter = GridTools::minimal_cell_diameter(triangulation);
	if (distort_factor > 0.0) {
		if (dim < 3)
			GridTools::distort_random(distort_factor, triangulation);
		else
			triangulation.begin_active()->vertex(6) = triangulation.begin_active()->vertex(6)
												      + Point<dim>(distort_factor*diameter,
												    		       distort_factor*diameter,
																   distort_factor*diameter);
	}

	// Set boundary indicator

	if (dim == 1) return;

	// Top and bottom faces are Neumann (indicator = 1), otherwise Dirichlet (indicator = 0)
	typename Triangulation<dim>::cell_iterator
	cell = triangulation.begin(),
	endc = triangulation.end();
	for ( ; cell!=endc ; cell++) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if ((std::fabs(cell->face(face)->center()(1)) < diameter/10.0) ||
				(std::fabs(cell->face(face)->center()(1) - 1.0) < diameter/10.0))
				cell->face(face)->set_all_boundary_ids(1);
			if (dim == 3) {
				if ((std::fabs(cell->face(face)->center()(2)) < diameter/10.0) ||
					(std::fabs(cell->face(face)->center()(2) - 1.0) < diameter/10.0))
					cell->face(face)->set_all_boundary_ids(1);
			}
		}
	}
}


template <int dim>
void make_grid_network (Triangulation<dim>& triangulation,
						int global_refinement = 4)
{
	Assert(dim == 2, ExcNotImplemented());
	
	std::vector<unsigned int> repetitions(2);
	repetitions[0] = 7;
	repetitions[1] = 6;
	Point<dim> p1, p2;
	p2[0] = 700;
	p2[1] = 600;
	// Create mesh with dx = 100m (7x6 uniform mesh)
	GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);
	triangulation.refine_global(global_refinement);
	
	// Top and bottom faces are Neumann (indicator = 1), otherwise Dirichlet (indicator = 0)
	typename Triangulation<dim>::cell_iterator
	cell = triangulation.begin(),
	endc = triangulation.end();
	for ( ; cell!=endc ; cell++) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if (cell->at_boundary(face) && (face == 2 || face == 3) )
				cell->face(face)->set_all_boundary_ids(1);
		}
	}
}


template <int dim>
void make_grid_flemisch(Triangulation<dim>& triangulation, unsigned int N, bool lr = false)
{
	Assert(dim == 2, ExcNotImplemented());
	
	if (lr) {
		Point<dim> p1;
		Point<dim> p2;
		for (unsigned int d=0; d<dim; ++d)
			p2[d] = 1.0;
		const double H = 1.0/(N-1);
		double h = (1.0-H)/(N-1);
		std::vector<double> step_size(N, h);
		step_size[0] = H;
		std::vector<std::vector<double>> step_sizes(dim, step_size);
		GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, p1, p2, false);
	}
	else {
		Assert(N % 2 == 1, ExcMessage("N should be odd to avoid element faces along fractures."));
		GridGenerator::subdivided_hyper_cube(triangulation, N);
	}

	const double diameter = GridTools::minimal_cell_diameter(triangulation);

	// Top and bottom faces are Neumann (indicator = 1), otherwise Dirichlet (indicator = 0)
	typename Triangulation<dim>::cell_iterator
	cell = triangulation.begin(),
	endc = triangulation.end();
	for ( ; cell!=endc ; cell++) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if ((std::fabs(cell->face(face)->center()(1)) < diameter/10.0) ||
				(std::fabs(cell->face(face)->center()(1) - 1.0) < diameter/10.0))
				cell->face(face)->set_all_boundary_ids(1);
			else if (std::fabs(cell->face(face)->center()(0)) < diameter/10.0)
				cell->face(face)->set_all_boundary_ids(1);
		}
	}
}


template <int dim>
void make_grid_single_frac(Triangulation<dim>& triangulation);

template <>
void make_grid_single_frac(Triangulation<3>&)
{
	Assert(false, ExcNotImplemented());
}

template <>
void make_grid_single_frac(Triangulation<2>& triangulation)
{
	// Fracture width
	const double w = 1e-3;
	// Discretization level
	const unsigned int N = 1000;
	// Start and end point for fracture
	const Point<2> a(0.1, 0.25);
	const Point<2> b(1.0, 19.0/32.0);
	
	const double tol = 1e-12;
	
	Assert( std::abs(std::round(a(0)/(1.0/N)) - a(0)/(1.0/N) < tol), ExcMessage("a(0) should be equal to mesh line"));
	
	const double f = 1.25;
	const unsigned int N1 = (int) std::ceil(f*a(1)*N);
	const unsigned int N2 = N - N1 - 1;
	
	const double tantheta = (b(1)-a(1))/(b(0)-a(0));
	const double costheta = cos(atan(tantheta));
	
	const double hf = w*costheta;
	std::vector<double> dx(N, 1.0/N);
	
	std::vector<double> dy(N, (1.0 - (a(1)+hf/2.0))/N2);
	const double delta = (a(1)-hf/2.0)/N1;
	for (unsigned int i=0; i<N1; ++i) {
		dy[i] = delta;
	}
	dy[N1] = hf;
	std::vector<std::vector<double>> step_sizes;
	step_sizes.push_back(dx);
	step_sizes.push_back(dy);
	
	const double d = 0.5*hf*tantheta;
	
	GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, Point<2>(0.0, 0.0), Point<2>(1.0,1.0), false);
	
	// Top and bottom faces are Neumann (indicator = 1), otherwise Dirichlet (indicator = 0)
	typename Triangulation<2>::cell_iterator
	cell = triangulation.begin(),
	endc = triangulation.end();
	for ( ; cell!=endc ; cell++) {
		for (unsigned int face=2; face<GeometryInfo<2>::faces_per_cell; ++face) {
			if (cell->at_boundary(face))
				cell->face(face)->set_all_boundary_ids(1);
		}
	}
	
	// Iterate vertices
	typename Triangulation<2>::vertex_iterator vi = triangulation.begin_vertex();
	for (; vi != triangulation.end_vertex(); ++vi) {
		Point<2> vertex = vi->vertex();
		
		// Move x coordinate
		if ( a(0)-hf/2.0 < vertex(0) && a(0)+hf/2.0 > vertex(0) ) {
			if (vertex(1) > a(1))
				vertex(0) = a(0) - d;
			else
				vertex(0) = a(0) + d;
		}
		else if (vertex(0) > a(0)-hf/2.0 && vertex(0) < 1.0-tol) {
			if (vertex(1) > a(1))
				vertex(0) = vertex(0) - d;
			else
				vertex(0) = vertex(0) + d;
		}
		
		// Move y coordinate
		if (vertex(0) > a(0)+hf/2.0) {
			double x1, x2, y1, y2;
			x2 = 1.0;
			y1 = vertex(1);
			if ( abs(vertex(1) - (a(1) - hf/2.0)) < tol ) {
				x1 = a(0) + d;
				y2 = b(1) - hf/2.0;
			}
			else if ( abs(vertex(1) - (a(1) + hf/2.0)) < tol ) {
				x1 = a(0) - d;
				y2 = b(1) + hf/2.0;
			}
			else if (vertex(1) < a(1) - hf) {
				x1 = a(0) + d;
				y2 = (b(1)-hf/2.0)*vertex(1)/(a(1)-hf/2.0);
			}
			else {
				x1 = a(0) - d;
				y2 = (1.0-(b(1)+hf/2.0)) * (vertex(1)-(a(1)+hf/2.0))/(1-(a(1)+hf/2.0)) + b(1)+hf/2.0;
			}
			vertex(1) = (y2-y1)/(x2-x1) * (vertex(0)-x1) + vertex(1);
		}
		
		// Actually update vertex
		vi->vertex() = vertex;
	}
	
	// Refine 3 times inside fracture
	for (unsigned int i=0; i<3; ++i) {
		typename Triangulation<2>::active_cell_iterator
		cell = triangulation.begin_active(),
		endc = triangulation.end();
		for ( ; cell!=endc ; cell++) {
			const Point<2> center = cell->center();
			const double y_frac = tantheta * (center(0)-a(0)) + a(1);
			if (center(0) > a(0) && abs(center(1)-y_frac) < hf/2.0)
				cell->set_refine_flag();
		}
		triangulation.execute_coarsening_and_refinement();
	}
}


template<int dim>
void refine_global_anisotropic(Triangulation<dim>& triangulation, int dir)
{
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation.begin_active(),
	endc = triangulation.end();
	for ( ; cell!=endc ; cell++) {
		cell->set_refine_flag(RefinementCase<dim>::cut_axis(dir));
	}
	triangulation.execute_coarsening_and_refinement();
}
	

template <int dim>
void make_grid_1D (Triangulation<dim>& triangulation,
				   unsigned int refinement = 2,
				   bool uniform = true)
{
	Assert(dim == 2, ExcInternalError());

	GridGenerator::hyper_cube(triangulation, 0, 1);
	if (uniform) {
		for (unsigned int i=0; i<refinement; ++i)
			refine_global_anisotropic(triangulation, 0);
	}
	else {
		for (unsigned int i=0; i<refinement; ++i) {
			typename Triangulation<dim>::cell_iterator cell = triangulation.begin();
			while( ! cell->active())
				cell = cell->child(0);
			cell->set_refine_flag(RefinementCase<dim>::cut_axis(0));
			triangulation.execute_coarsening_and_refinement();
		}
	}

	// Top and bottom faces are Neumann (indicator = 1), otherwise Dirichlet (indicator = 0)
	const double diameter = GridTools::minimal_cell_diameter(triangulation);
	typename Triangulation<dim>::cell_iterator
	cell = triangulation.begin(),
	endc = triangulation.end();
	for ( ; cell!=endc ; cell++) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if ((std::fabs(cell->face(face)->center()(1)) < diameter/10.0) ||
				(std::fabs(cell->face(face)->center()(1) - 1.0) < diameter/10.0))
				cell->face(face)->set_all_boundary_ids(1);
			if (dim == 3) {
				if ((std::fabs(cell->face(face)->center()(2)) < diameter/10.0) ||
					(std::fabs(cell->face(face)->center()(2) - 1.0) < diameter/10.0))
					cell->face(face)->set_all_boundary_ids(1);
			}
		}
	}
}


template <int dim_st>
void make_grid_spacetime(Triangulation<dim_st>& triangulation,
						  std::vector<unsigned int> element_repititions_space,
						  double t_end,
						  unsigned int n_time_steps)
{
	AssertDimension(element_repititions_space.size(), dim_st-1);
	Point<dim_st> p0;
	Point<dim_st> p1;
	p1(dim_st-1) = t_end;
	for (unsigned int d=0; d<dim_st-1; ++d)
		p1(d) = 1.0;
	element_repititions_space.push_back(n_time_steps);
	GridGenerator::subdivided_hyper_rectangle(triangulation, element_repititions_space, p0, p1);
	
	// Use subdomain_id to identify time slab.
	// With GridGenerator::subdivided_hyper_rectangle(...), cells are ordered by last dimension (time)
	unsigned int cells_per_slab = 1;
	for (unsigned int i=0; i<element_repititions_space.size() -1; ++i)
		cells_per_slab *= element_repititions_space[i];
	unsigned int subdomain = 0;
	unsigned int cell_count = 0;
	
	typename Triangulation<dim_st>::cell_iterator
	cell = triangulation.begin(),
	endc = triangulation.end();
	for ( ; cell!=endc; ++cell) {
		if (cell_count >= cells_per_slab) {
			++subdomain;
			cell_count = 0;
		}
		cell->set_subdomain_id(subdomain);
		++cell_count;
	}
}


template <int dim_st>
void make_grid_spacetime(Triangulation<dim_st>& triangulation,
						  unsigned int element_repititions_space,
						  double t_end,
						  unsigned int n_time_steps)
{
	std::vector<unsigned int> element_repititions(dim_st-1, element_repititions_space);
	make_grid_spacetime(triangulation, element_repititions, t_end, n_time_steps);
}


template <int dim_st>
void make_grid_spacetime_network(Triangulation<dim_st>& triangulation,
								 unsigned int initial_refinement_space,
								 double t_end,
								 unsigned int n_time_steps)
{
	Assert(dim_st == 3, ExcNotImplemented());
	Point<dim_st> p0;
	Point<dim_st> p1;
	p1[0] = 700;
	p1[1] = 600;
	p1(2) = t_end;
	std::vector<unsigned int> repetitions(3);
	repetitions[0] = 7 * pow(2,initial_refinement_space);
	repetitions[1] = 6 * pow(2,initial_refinement_space);
	repetitions[2] = n_time_steps;
	GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p0, p1);
	
	// Use subdomain_id to identify time slab.
	// With GridGenerator::subdivided_hyper_rectangle(...), cells are ordered by last dimension (time)
	unsigned int cells_per_slab = 1;
	for (unsigned int i=0; i<repetitions.size() -1; ++i)
		cells_per_slab *= repetitions[i];
	unsigned int subdomain = 0;
	unsigned int cell_count = 0;
	
	typename Triangulation<dim_st>::cell_iterator
	cell = triangulation.begin(),
	endc = triangulation.end();
	for ( ; cell!=endc; ++cell) {
		if (cell_count >= cells_per_slab) {
			++subdomain;
			cell_count = 0;
		}
		cell->set_subdomain_id(subdomain);
		++cell_count;
	}
}


void make_grid_1D (Triangulation<2>& triangulation,
				   double h1)
{
	const int dim = 2;
	// See step 14
	
	std::vector<Point<dim>> vertices;
	vertices.push_back(Point<dim>(0.0, 0.0));
	vertices.push_back(Point<dim>(h1,  0.0));
	vertices.push_back(Point<dim>(1.0, 0.0));
	vertices.push_back(Point<dim>(0.0, 1.0));
	vertices.push_back(Point<dim>(h1,  1.0));
	vertices.push_back(Point<dim>(1.0, 1.0));
	
	const int n_cells = 2;
	static const int cell_vertices[][GeometryInfo<dim>::vertices_per_cell] = {{0, 1, 3, 4}, {1, 2, 4, 5}};
	std::vector<CellData<dim>> cells(n_cells, CellData<dim>());
	for (unsigned int i=0; i<n_cells; ++i) {
		for (unsigned int j=0; j<GeometryInfo<dim>::vertices_per_cell; ++j)
			cells[i].vertices[j] = cell_vertices[i][j];
		cells[i].material_id = 0;
	}
	
	triangulation.create_triangulation (vertices, cells, SubCellData());
	
	// Set boundary indicator
	if (dim == 1) return;

	// Top and bottom faces are Neumann (indicator = 1), otherwise Dirichlet (indicator = 0)
	const double diameter = GridTools::minimal_cell_diameter(triangulation);
	typename Triangulation<dim>::cell_iterator
	cell = triangulation.begin(),
	endc = triangulation.end();
	for ( ; cell!=endc ; cell++) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if ((std::fabs(cell->face(face)->center()(1)) < diameter/10.0) ||
				(std::fabs(cell->face(face)->center()(1) - 1.0) < diameter/10.0))
				cell->face(face)->set_all_boundary_ids(1);
			//if (std::fabs(cell->face(face)->center()(0)) > 1.0 - diameter/10.0)
			//	cell->face(face)->set_boundary_indicator(1);
			if (dim == 3) {
				if ((std::fabs(cell->face(face)->center()(2)) < diameter/10.0) ||
					(std::fabs(cell->face(face)->center()(2) - 1.0) < diameter/10.0))
					cell->face(face)->set_all_boundary_ids(1);
			}
		}
	}
}


void make_grid_2D (Triangulation<2>& triangulation, double h1)
{
	const int dim = 2;

	/*
	GridGenerator::hyper_cube(triangulation, 0, 1);
	triangulation.refine_global();
	triangulation.begin_active()->set_refine_flag();
	triangulation.execute_coarsening_and_refinement();
	//GridTools::distort_random(0.4, triangulation);
	*/
	
	std::vector<Point<dim>> vertices;
	vertices.push_back(Point<dim>(0.0, 0.0));
	vertices.push_back(Point<dim>(h1,  0.0));
	vertices.push_back(Point<dim>(1.0, 0.0));
	vertices.push_back(Point<dim>(0.0, 1.0));
	vertices.push_back(Point<dim>(h1,  1.0));
	vertices.push_back(Point<dim>(1.0, 1.0));
	
	const int n_cells = 2;
	static const int cell_vertices[][GeometryInfo<dim>::vertices_per_cell] = {{0, 1, 3, 4}, {1, 2, 4, 5}};
	std::vector<CellData<dim>> cells(n_cells, CellData<dim>());
	for (unsigned int i=0; i<n_cells; ++i) {
		for (unsigned int j=0; j<GeometryInfo<dim>::vertices_per_cell; ++j)
			cells[i].vertices[j] = cell_vertices[i][j];
		cells[i].material_id = 0;
	}
	
	triangulation.create_triangulation (vertices, cells, SubCellData());
	(++(triangulation.begin_active()))->set_refine_flag();
	triangulation.execute_coarsening_and_refinement();
	//triangulation.refine_global();
	GridTools::distort_random(0.5, triangulation);
	
	//triangulation.begin_active()->set_refine_flag(RefinementCase<dim>::cut_axis(1));
	//triangulation.execute_coarsening_and_refinement();
	
	// Top and bottom faces are Neumann (indicator = 1), otherwise Dirichlet (indicator = 0)
	const double diameter = GridTools::minimal_cell_diameter(triangulation);		
	typename Triangulation<dim>::cell_iterator
	cell = triangulation.begin(),
	endc = triangulation.end();
	for ( ; cell!=endc ; cell++) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			cell->face(face)->set_all_boundary_ids(1);
			if ((std::fabs(cell->face(face)->center()(1)) < diameter/10.0) ||
				(std::fabs(cell->face(face)->center()(1) - 1.0) < diameter/10.0))
				cell->face(face)->set_all_boundary_ids(1);
			if (std::fabs(cell->face(face)->center()(0)) < 0.0 + diameter/10.0)
				cell->face(face)->set_all_boundary_ids(1);
			if (dim == 3) {
				if ((std::fabs(cell->face(face)->center()(2)) < diameter/10.0) ||
					(std::fabs(cell->face(face)->center()(2) - 1.0) < diameter/10.0))
					cell->face(face)->set_all_boundary_ids(1);
			}
		}
	}
}


template <int dim>
void refine_around_fractures(Triangulation<dim>& tria, FractureNetwork fractures, unsigned int ref_cycles = 1, bool ref_neighbors = false);

template <>
void refine_around_fractures(Triangulation<3>&, FractureNetwork, unsigned int, bool)
{
	Assert(false, ExcNotImplemented());
}

template <>
void refine_around_fractures(Triangulation<2>& tria, FractureNetwork fractures, unsigned int ref_cycles, bool ref_neighbors)
{
	typedef typename Triangulation<2>::active_cell_iterator ACI;
	for (unsigned int cycle=0; cycle<ref_cycles; ++cycle) {
		fractures.init_fractures(tria, 2);
		unsigned int n_cells_old = tria.n_active_cells();
		for (FractureNetwork::ConstIterator fracture = fractures.begin(); fracture != fractures.end(); ++fracture) {
			for (unsigned int vi=0; vi<fracture->n_vertices()-1; ++vi) {
				typename EmbeddedSurface<2>::IndexPair cell_info = fracture->get_segment_cell_info(vi);
				ACI cell(&tria, cell_info.first, cell_info.second);
				cell->set_refine_flag();
				if (ref_neighbors) {
					std::vector<ACI> neighbors;
					GridTools::get_active_neighbors<Triangulation<2>>(cell, neighbors);
					for (unsigned int ni=0; ni<neighbors.size(); ++ni)
						neighbors[ni]->set_refine_flag();
				}
			}
		}
		tria.execute_coarsening_and_refinement();
		std::cout << "Refined locally around fractures: " << n_cells_old << " -> " << tria.n_active_cells() << " cells." << std::endl;
	}
}


void get_active_vertex_neighbors(Triangulation<2>::active_cell_iterator cell, std::vector<Triangulation<2>::active_cell_iterator>& active_neighbors)
{
	typedef typename Triangulation<2>::active_cell_iterator ACI;

	ACI face_neighbor;
	ACI diag_neighbor;
	for (unsigned int face_x=0; face_x<2; ++face_x) {
		if (cell->at_boundary(face_x))
			continue;
		for (unsigned int face_y=2; face_y<4; ++face_y) {
			if (cell->face(face_x)->has_children())
				face_neighbor = cell->neighbor_child_on_subface(face_x, face_y-2);
			else
				face_neighbor = cell->neighbor(face_x);
			active_neighbors.push_back(face_neighbor);
			if (cell->neighbor(face_x)->at_boundary(face_y))
				continue;
			if (cell->neighbor_is_coarser(face_x)) {
				std::pair<unsigned int, unsigned int> neighbor_info = cell->neighbor_of_coarser_neighbor(face_x);
				if (face_y-neighbor_info.second != 2)
					continue;
			}
			if (face_neighbor->face(face_y)->has_children())
				diag_neighbor = face_neighbor->neighbor_child_on_subface(face_y, GeometryInfo<2>::opposite_face[face_x]);
			else
				diag_neighbor = face_neighbor->neighbor(face_y);
			
			active_neighbors.push_back(diag_neighbor);
		}
	}
}


void refine_around_close_fractures(Triangulation<2>& tria, FractureNetwork fractures, unsigned int max_ref_cycles = 10)
{
	typedef typename Triangulation<2>::active_cell_iterator ACI;
	
	bool done = false;
	unsigned int cycle = 0;
	
	while (!done && cycle < max_ref_cycles) {
		fractures.init_fractures(tria, 2);
		const unsigned int n_cells_old = tria.n_active_cells();
		unsigned int fi = 0;
		for (FractureNetwork::ConstIterator fracture_i = fractures.begin(); fracture_i != fractures.end(); ++fracture_i, ++fi) {
			
			const Point<2> a_i = fracture_i->vertex(0);
			const Point<2> b_i = fracture_i->vertex(fracture_i->n_vertices()-1);
			const double length_i = a_i.distance(b_i);
			
			// Flag cells that contain current fracture
			tria.clear_user_flags();
			for (unsigned int vi=0; vi<fracture_i->n_vertices()-1; ++vi) {
				typename EmbeddedSurface<2>::IndexPair cell_info = fracture_i->get_segment_cell_info(vi);
				ACI cell(&tria, cell_info.first, cell_info.second);
				cell->set_user_flag();
			}
			
			// Loop over all other fratures
			unsigned int fj = 0;
			for (FractureNetwork::ConstIterator fracture_j = fractures.begin(); fracture_j != fractures.end(); ++fracture_j, ++fj) {
				if (fi == fj)
					continue;
				
				const Point<2> a_j = fracture_j->vertex(0);
				const Point<2> b_j = fracture_j->vertex(fracture_j->n_vertices()-1);
				const double length_j = a_j.distance(b_j);
				
				// Find intersection point
				Point<2> x;
				bool intersect = line_intersection(a_i, b_i, a_j, b_j, x);
				if (a_i.distance(x) > length_i || b_i.distance(x) > length_i)
					intersect = false;
				else if (a_j.distance(x) > length_j || b_j.distance(x) > length_j)
					intersect = false;
				
				// If intersection, continue
				if (intersect)
					continue;
				
				for (unsigned int vj=0; vj<fracture_j->n_vertices()-1; ++vj) {
					typename EmbeddedSurface<2>::IndexPair cell_info = fracture_j->get_segment_cell_info(vj);
					ACI cell(&tria, cell_info.first, cell_info.second);
					
					if (cell->user_flag_set())
						cell->set_refine_flag();
					else {
						std::vector<ACI> patch;
						get_active_vertex_neighbors(cell, patch);
						
						for (unsigned int ci=0; ci<patch.size(); ++ci)
							if (patch[ci]->user_flag_set()) {
								//patch[ci]->set_refine_flag();
								cell->set_refine_flag();
							}
					}
				}
			}
		}
		
		tria.execute_coarsening_and_refinement();
		Assert(tria.n_active_cells() >= n_cells_old, ExcMessage("Mesh should not have been coarsened."));
		
		std::cout << "Refined around close fractures: " << n_cells_old << " -> " << tria.n_active_cells() << " cells." << std::endl;
		
		if (tria.n_active_cells() == n_cells_old)
			done = true;
		
		++cycle;
	}
	
	if (!done)
		std::cout << "Warning! Mesh not fully resolved around close fractures due to max ref cycles." << std::endl;
}


template <int dim>
void make_grid_flemisch_resolved(Triangulation<dim>& triangulation);

template <>
void make_grid_flemisch_resolved(Triangulation<3>&)
{
	Assert(false, ExcNotImplemented());
}

template <>
void make_grid_flemisch_resolved(Triangulation<2>& triangulation)
{
	const double w = 1e-4;
	const unsigned int N1 = 78;
	const unsigned int N2 = 19;
	const unsigned int N3 = 19;
	const unsigned int N4 = 39;
	
	const double d1 = 63.0/2.0*w;
	const double d2 = 65.0/2.0*w;
	
	std::vector<double> dx(N1 + N2 + N3 + N4 + 3);
	
	unsigned int i=0;
	for (; i<N1; ++i)
		dx[i] = (0.5-d1)/N1;
	dx[i++] = 64.0*w;
	for (; i<N1+1+N2; ++i)
		dx[i] = ((0.625-d1)-(0.5+d2)) / N2;
	dx[i++] = 64.0*w;
	for (; i<N1+1+N2+1+N3; ++i)
		dx[i] = ((0.75-d1)-(0.625+d2)) / N3;
	dx[i++] = 64.0*w;
	for (; i<N1+1+N2+1+N3+1+N4; ++i)
		dx[i] = (1.0-(0.75+d2)) / N4;
	
	const std::vector<std::vector<double>> step_sizes(2, dx);
	GridGenerator::subdivided_hyper_rectangle(triangulation, step_sizes, Point<2>(0.0, 0.0), Point<2>(1.0,1.0), false);
	
	// Set pressure boundary conditions
	std::vector<unsigned int> boundary_ids = get_boundary_id(FLEMISCH_RESOLVED);
	for (typename Triangulation<2>::active_cell_iterator cell = triangulation.begin_active(); cell != triangulation.end(); ++cell) {
		for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f) {
			if (cell->at_boundary(f))
				cell->face(f)->set_boundary_id(boundary_ids[f]);
		}
	}
	
	// Refine around fractures
	std::vector<double> delta(6);
	delta[0] = 32*w;
	delta[1] = 16*w;
	delta[2] = 8*w;
	delta[3] = 4*w;
	delta[4] = 2*w;
	delta[5] = 1*w;
	for (unsigned int r=0; r<6; ++r) {
		unsigned int n_cells_old = triangulation.n_active_cells();
		const double d = delta[r]  - 0.1*w;
		for (typename Triangulation<2>::active_cell_iterator cell = triangulation.begin_active(); cell != triangulation.end(); ++cell) {
			const Point<2> cc = cell->center();
			const double x = cc[0];
			const double y = cc[1];
			
			if ( y > 0.5-d && y < 0.5+d )
				cell->set_refine_flag();
			else if ( x > 0.5-d && x < 0.5+d )
				cell->set_refine_flag();
			else if ( x > 0.5 && (y > 0.75-d && y < 0.75+d) )
				cell->set_refine_flag();
			else if ( y > 0.5 && (x > 0.75-d && x < 0.75+d) )
				cell->set_refine_flag();
			else if ( (x > 0.5 && x < 0.75) && (y > 0.625-d && y < 0.625+d) )
				cell->set_refine_flag();
			else if ( (y > 0.5 && y < 0.75) && (x > 0.625-d && x < 0.625+d) )
				cell->set_refine_flag();
		}
		triangulation.execute_coarsening_and_refinement();
		std::cout << "Refined locally around fractures: " << n_cells_old << " -> " << triangulation.n_active_cells() << " cells." << std::endl;
	}
	
	std::cout << "Min h: " << GridTools::minimal_cell_diameter(triangulation) << std::endl;
	
	//FractureNetwork fractures(FLEMISCH);
	//refine_around_fractures(triangulation, fractures, 6, false);
}


template <int dim>
void set_linear_pressure_bcs(Triangulation<dim> &tria, ProblemFunctionsFlow<dim> &flow_fun, unsigned int dir = 0, double magnitude = 1.0)
{
	flow_fun.set_linear_pressure(dir, tria, magnitude);
	typename Triangulation<dim>::cell_iterator
	cell = tria.begin(),
	endc = tria.end();
	for ( ; cell!=endc ; cell++) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if (cell->at_boundary(face)) {
				if ( GeometryInfo<dim>::unit_normal_direction[face] == dir)
					cell->face(face)->set_all_boundary_ids(0);
				else
					cell->face(face)->set_all_boundary_ids(1);
			}
		}
	}
}


template <int dim>
void set_well_bcs(Triangulation<dim> &tria, ProblemFunctionsFlow<dim> &flow_fun, ProblemFunctionsTransport<dim> &flow_transport)
{
	flow_fun.right_hand_side = new WellFunction<dim>(0.025, 100.0);
	flow_transport.right_hand_side = new WellFunction<dim>(0.025, 100.0);
	flow_fun.pure_neumann = true;
	// Set all boundary to Neumann
	typename Triangulation<dim>::cell_iterator
	cell = tria.begin(),
	endc = tria.end();
	for ( ; cell!=endc ; cell++) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if (cell->at_boundary(face)) {
				cell->face(face)->set_all_boundary_ids(1);
			}
		}
	}
}


/* Calculate L2 norm of face-constant property.
 * The deal II function integrate_difference() does not work on faces.
 */
template <int dim>
double L2_norm_face(Triangulation<dim>& triangulation, const DoFHandler<dim>& dh_face, Vector<double> vec)
{
	triangulation.clear_user_flags();
	double norm = 0.0;

	typename DoFHandler<dim>::active_cell_iterator
	cell = dh_face.begin_active(),
	endc = dh_face.end();
	for ( ; cell!=endc; ++cell) {
		std::vector<unsigned int> face_dofs(GeometryInfo<dim>::faces_per_cell);
		cell->get_dof_indices(face_dofs);
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if ( cell->at_boundary(face) ) {
				norm += pow(vec(face_dofs[face]), 2) * face_measure<dim>(cell, face);
				continue;
			}
			// If neighbor not active, skip this face, since neighbor has children
			// Instead visit this face from the finer cell later
			if ( ! cell->neighbor(face)->active() ) {
				continue;
			}
			const typename DoFHandler<dim>::active_cell_iterator neighbor = cell->neighbor(face);
			// If neighbor not visited or neighbor is coarser
			if ( ( ! neighbor->user_flag_set() ) || ( cell->neighbor_is_coarser(face) ) ) {
				norm += pow(vec(face_dofs[face]), 2) * face_measure<dim>(cell, face);
			}
		}
		cell->set_user_flag();
	}
	triangulation.clear_user_flags();
	norm = sqrt(norm);
	return norm;
}


// Simply prints header to screen
void print_header(const char* header)
{
	std::cout << std::endl;
	std::cout << "### " << header << " ###" << std::endl;
	std::cout << std::endl;
}


/* Calculate edge norm of face-constant property (see definition in Larson & Niklasson 2004).
 * The deal II function integrate_difference() does not work on faces.
 */
template <int dim>
double edgenorm(Triangulation<dim>& triangulation, const DoFHandler<dim>& dh_face, Vector<double> vec)
{
	triangulation.clear_user_flags();
	double norm = 0.0;

	typename DoFHandler<dim>::active_cell_iterator
	cell = dh_face.begin_active(),
	endc = dh_face.end();
	for ( ; cell!=endc; ++cell) {
		std::vector<unsigned int> face_dofs(GeometryInfo<dim>::faces_per_cell);
		cell->get_dof_indices(face_dofs);
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if ( cell->at_boundary(face) ) {
				norm += pow(vec(face_dofs[face]) * face_measure<dim>(cell, face), 2);
				continue;
			}
			// If neighbor not active, skip this face, since neighbor has children
			// Instead visit this face from the finer cell later
			if ( ! cell->neighbor(face)->active() ) {
				continue;
			}
			const typename DoFHandler<dim>::active_cell_iterator neighbor = cell->neighbor(face);
			// If neighbor not visited or neighbor is coarser
			if ( ( ! neighbor->user_flag_set() ) || ( cell->neighbor_is_coarser(face) ) ) {
				norm += pow(vec(face_dofs[face]) * face_measure<dim>(cell, face), 2);
			}
		}
		cell->set_user_flag();
	}
	triangulation.clear_user_flags();
	norm = sqrt(norm);
	return norm;
}


// Overload << for std::vector
template <class type>
std::ostream& operator<<(std::ostream& os, const std::vector<type> vec)
{
	typename std::vector<type>::const_iterator
	it    = vec.begin(),
	endit = vec.end();
	os << "[";
	bool first = true;
	for ( ; it != endit; ++it ) {
		if (!first)
			os << ", ";
		else
			first = false;
		os << *it;
	}
	os << "]" << std::endl;
	return os;
}


Vector<double> copy_vector(const std::vector<double> vec_in)
{
	return Vector<double>(vec_in.begin(), vec_in.end());
}


// Reinit vector (sequential mode)
void reinit_vec_seq(Vector<double>& v, unsigned int n)
{
	v.reinit(n);
}

void reinit_vec_seq(TrilinosWrappers::MPI::Vector& v, unsigned int n)
{
	v.reinit(complete_index_set(n));
}


// Setup flux constraints so that sum flux is equal over subfaces for grids with hanging nodes.
// Set dofs_integrated to false if flux dofs are not integrated (dofs_integrated = true for RT0)
template <int dim>
void setup_flux_constraints_subfaces(const DoFHandler<dim>& dh, ConstraintMatrix& constraints, bool dofs_integrated = true)
{
	const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
	Assert(faces_per_cell == dh.get_fe().dofs_per_cell, ExcMessage("Dofs per cell should equal faces per cell."));
	typename DoFHandler<dim>::active_cell_iterator
	cell = dh.begin_active(),
	endc = dh.end();
	for ( ; cell != endc; ++ cell) {
		for (unsigned int face=0; face<faces_per_cell; ++ face) {
			// If face has children, set this dof equal to sum of dofs on finer faces.
			if ( cell->face(face)->has_children() ) {
				std::vector<unsigned int> cell_dofs(faces_per_cell);
				cell->get_dof_indices(cell_dofs);
				const unsigned int line = cell_dofs[face];
				constraints.add_line(line);
				// Loop through subfaces and add entries to constraint line
				for (unsigned int subface=0; subface<GeometryInfo<dim>::max_children_per_face; ++subface) {
					const typename DoFHandler<dim>::active_cell_iterator
						neighbor = cell->neighbor_child_on_subface(face, subface);
					std::vector<unsigned int> neighbor_dofs(faces_per_cell);
					neighbor->get_dof_indices(neighbor_dofs);
					const unsigned int neighbor_face = GeometryInfo<dim>::opposite_face[face];
					double entry = 1.0;
					if ( !dofs_integrated)
						entry = face_measure<dim>(neighbor, neighbor_face) / face_measure<dim>(cell, face);
					constraints.add_entry(line, neighbor_dofs[neighbor_face], entry);
				}
			}
		}
	}
}


#endif // HELP_FUNCTIONS_H
