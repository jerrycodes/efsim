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

#ifndef EMBEDDED_SURFACE_H
#define EMBEDDED_SURFACE_H

#include "ProblemFunctions.h"

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>

using namespace dealii;


// Classes for representing embedded surfaces


template <int dim>
struct Intersection
{
	Intersection(unsigned int vi,
				 typename Triangulation<dim>::active_cell_iterator cell,
				 unsigned int face)
	: n_cells(1), index(vi), cells(n_cells), faces(n_cells)
	{
		cells[0] = IndexPair(cell->level(), cell->index());
		faces[0] = face;
	}
	
	Intersection(unsigned int vi,
				 typename Triangulation<dim>::active_cell_iterator cell1,
				 typename Triangulation<dim>::active_cell_iterator cell2,
				 unsigned int face1,
			     unsigned int face2)
	: n_cells(2), index(vi), cells(n_cells), faces(n_cells)
	{
		cells[0] = IndexPair(cell1->level(), cell1->index());
		cells[1] = IndexPair(cell2->level(), cell2->index());
		faces[0] = face1;
		faces[1] = face2;
	}
	
	typedef std::pair<unsigned int, unsigned int> IndexPair;
	
	unsigned int n_cells;
	unsigned int index;
	std::vector<IndexPair> cells;
	std::vector<unsigned int> faces;
};


// Forward declarations

template <int dim>
unsigned int boundary_intersection(Point<dim>&, typename Triangulation<dim>::active_cell_iterator, Point<dim>, Point<dim>);

template <int dim>
typename Triangulation<dim>::active_cell_iterator find_active_neighbor_on_face(typename Triangulation<dim>::active_cell_iterator cell, 
																			   const unsigned int face, const Point<dim> x);

template <int dim>
typename Triangulation<dim>::active_cell_iterator find_active_cell_around_vertex(typename Triangulation<dim>::active_cell_iterator cell, 
																				 const unsigned int vertex_i, const Point<dim> x);

template <int dim>
int on_cell_boundary(const typename Triangulation<dim>::active_cell_iterator cell, const Point<dim> p);


// Model surface as a polygonal chain
template <int dim>
class EmbeddedSurface
{
public:
	EmbeddedSurface(std::vector<Point<dim>> v);
	EmbeddedSurface(Point<dim> a, Point<dim> b, unsigned int n = 1);
	EmbeddedSurface(FractureParametrization* fracture, double approx_surface_length, unsigned int n = 1);
	
	void setup(const Triangulation<dim>& tria, unsigned int n_qpoints = 2);
	Vector<double> initialize(const Triangulation<dim>& tria);
	void linearize(unsigned int n);
	void double_precision();
	
	unsigned int n_vertices() const
	{ return vertices.size(); }
	Point<dim> vertex(unsigned int i) const;
	
	typedef std::pair<unsigned int, unsigned int> IndexPair;
	IndexPair get_segment_cell_info(unsigned int i) const;
	
	typename std::vector<Intersection<dim>>::const_iterator begin_intersection() const
	{ return intersections.begin(); }
	typename std::vector<Intersection<dim>>::const_iterator end_intersection() const
	{ return intersections.end(); }
	
	Intersection<dim> first_intersection() const
	{ return intersections.front(); }
	Intersection<dim> last_intersection() const
	{ return intersections.back(); }
	
	void print_to_screen(const Triangulation<dim>& tria) const;
	void output_to_vtk(std::string file_base) const;
	void output_to_vtk(std::string file_base, const Triangulation<dim>& tria) const;
	
private:
	// Storing points defining the polygonal chain
	// TODO: Can we use list?
	std::vector<Point<dim>> vertices;
	std::vector<IndexPair> segment_to_cell;
	unsigned int n_vertices_pre_initialize;
	double surface_length;
	unsigned int segments_per_cell = 2;
	
	std::vector<Intersection<dim>> intersections;
	void add_vertex_intersection(unsigned int vi, Point<dim> p1, Point<dim> x,
								 typename Triangulation<dim>::active_cell_iterator cell1,
								 typename Triangulation<dim>::active_cell_iterator cell2);
	
	typedef typename std::vector<Point<dim>>::iterator VertexIterator;
	
	enum SurfaceType { STRAIGHT, PARAMETRIZATION, POLYGON };
	SurfaceType surface_type;
	
	// Parametrization of curve
	FractureParametrization* parametrization;
	
	void linearize_parametrization(unsigned int n);
	void linearize_line(unsigned int n);
};


// Take list of vertices as input
template <int dim>
EmbeddedSurface<dim>::EmbeddedSurface(std::vector<Point<dim>> v)
: vertices(v), n_vertices_pre_initialize(v.size()), surface_type(POLYGON)
{
	surface_length = 0.0;
	for (unsigned int i=1; i<n_vertices(); ++i) {
		surface_length += vertices[i-1].distance(vertices[i]);
	}
}


// Straight surface from a to b with n segments
template <int dim>
EmbeddedSurface<dim>::EmbeddedSurface(Point<dim> a, Point<dim> b, unsigned int n)
: vertices(n+1), n_vertices_pre_initialize(n+1), surface_type(STRAIGHT)
{
	Assert(a.distance(b) > 1e-15, ExcMessage("Start and end point should not coincide."));
	surface_length = a.distance(b);
	const Tensor<1,dim> delta = (b-a)/n;
	vertices.front() = a;
	vertices.back() = b;
	for (unsigned int i=1; i<n; ++i)
		vertices[i] = a + i*delta;
}


template <int dim>
EmbeddedSurface<dim>::EmbeddedSurface(FractureParametrization* fracture, double approx_surface_length, unsigned int n)
:  n_vertices_pre_initialize(n+1), surface_length(approx_surface_length), surface_type(PARAMETRIZATION)
{
	parametrization = fracture;
	linearize_parametrization(n);
}


template <int dim>
void EmbeddedSurface<dim>::linearize(unsigned int n)
{
	n_vertices_pre_initialize = n+1;
	switch (surface_type)
	{
		case STRAIGHT:
			linearize_line(n);
			break;
		case PARAMETRIZATION:
			linearize_parametrization(n);
			break;
		default:
			Assert(false, ExcNotImplemented());
	}
}


template <int dim>
void EmbeddedSurface<dim>::linearize_line(unsigned int n)
{
	Assert(surface_type == STRAIGHT, ExcInternalError());
	Point<dim> a = vertices.front();
	Point<dim> b = vertices.back();
	vertices.resize(n+1);
	vertices.front() = a;
	vertices.back() = b;
	const Tensor<1,dim> delta = (b-a)/n;
	for (unsigned int i=1; i<n; ++i)
		vertices[i] = a + i*delta;
}

template <int dim>
void EmbeddedSurface<dim>::linearize_parametrization(unsigned int n)
{
	Assert(surface_type == PARAMETRIZATION, ExcInternalError());
	vertices.resize(n+1);
	const Point<1> s0_p(parametrization->start());
	const Point<1> s1_p(parametrization->end());
	Assert(s0_p.distance(s1_p) > 1e-15, ExcMessage("Start and end point should not coincide."));
	const Point<1> delta_s((s1_p-s0_p)/n);
	Point<1> s = s0_p;
	vertices.front()[0] = parametrization->value(s0_p,0);
	vertices.front()[1] = parametrization->value(s0_p,1);
	vertices.back()[0]  = parametrization->value(s1_p,0);
	vertices.back()[1]  = parametrization->value(s1_p,1);
	for (unsigned int vi=1; vi<n; ++vi) {
		s += delta_s;
		vertices[vi][0] = parametrization->value(s,0);
		vertices[vi][1] = parametrization->value(s,1);
		Assert(vertices[vi].distance(vertices[vi-1]) > 1e-15, ExcMessage("Neighboring points should not coincide."));
	}
}


template <int dim>
void EmbeddedSurface<dim>::setup(const Triangulation<dim>& tria, unsigned int n_qpoints)
{
	const double h_min = GridTools::minimal_cell_diameter(tria)/sqrt(dim);
	segments_per_cell = n_qpoints;
	if (surface_type != POLYGON)
		linearize((int)ceil(surface_length/h_min)*segments_per_cell);
}


template <int dim>
void EmbeddedSurface<dim>::double_precision()
{
       linearize(2*n_vertices_pre_initialize);
}


// Process vertices such that
// - Distance between two consecutive vertices are smaller than min_cell_diamter/2
// - Every pair of consecutive vertices belongs to the same cell (find cell boundary intersection and add to list of vertices)
// - Store the cell which each segment belongs to
template <int dim>
Vector<double> EmbeddedSurface<dim>::initialize(const Triangulation<dim>& tria)
{
	// Vector to store fracture_length_per_cell
	Vector<double> fracture_length_per_cell(tria.n_active_cells());
	
	// TODO: Handle the case when a segment is on a face. Use std::vector<bool>?
	const double L = sqrt(GridTools::volume(tria));
	const double tol = 1e-13;
	
	StaticMappingQ1<dim> mapQ1;
	
	const double h_min = GridTools::minimal_cell_diameter(tria)/sqrt(dim);
	const double max_segment_length = h_min / segments_per_cell;
	
	intersections.clear();
	segment_to_cell.clear();
	segment_to_cell.reserve(2*n_vertices());
	
	VertexIterator a = vertices.begin();
	VertexIterator b = a+1;
	bool previous_segment_intersected = false;
	unsigned int vi = 0;
	typename Triangulation<dim>::active_cell_iterator cell = GridTools::find_active_cell_around_point(tria, *a);
	int intersection_face = on_cell_boundary(cell, *a);
	if (intersection_face > -1) {
		intersections.push_back(Intersection<dim>(vi, cell, intersection_face));
		previous_segment_intersected = true;
	}
	Point<dim> a_prev = *a, b_prev = *b, cc_prev = cell->center(); a_prev[0] -= 1.0;
	while (b != vertices.end()) {
		//std::cout << *a << "   " << *b << " (Current cell center: " << cell->center() << ")" << std::endl;
		
		Assert(a->distance(*b) > tol, ExcMessage("Repeated points not allowed. Should not end up here."));
		
		if (! cell->point_inside(*a)) {
			std::cout << "Warning! Point a was not contained in given cell!" << std::endl
					  << "Should avoid ending here, since now we need to search globally for cell" << std::endl;
			cell = GridTools::find_active_cell_around_point(tria, *a);
		}
		
		if ( (*a == a_prev && *b==b_prev) && cc_prev == cell->center() ) {
			std::cout << "Error! Infinite loop. Exit" << std::endl;
			exit(1);
		}
		a_prev = *a, b_prev = *b, cc_prev = cell->center();
		
		// First, check if max lenght is satisfed. If not add midpoint.
		if (b->distance(*a) > max_segment_length) {
			const Point<dim> c((*a+*b)/2.0);
			b = vertices.insert(b, c);
			a = b-1; // Need to set a again since vector is updated
			continue;
		}
		
		// Boolean done tells if segment a and b lies in same cell
		bool done = cell->point_inside(*b);
		
		if (! done) {
			// First, check if we are at (or close to) a vertex
			// If so, update cell to the neighbor containing b, and set a = vertex
			for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v) {
				if (a->distance(cell->vertex(v)) < tol) {
					// Set a equals to vertex
					*a = cell->vertex(v);
					// Search for neighboring cell containing point b
					typename Triangulation<dim>::active_cell_iterator cell_prev = cell;
					cell = find_active_cell_around_vertex(cell, v, *b);
					done = true;
					
					// Add two intersections here (only if this is not an intersection point from previous iteration
					if (!previous_segment_intersected) {
						VertexIterator p = a-1;
						add_vertex_intersection(vi, *p, *a, cell_prev, cell);
					}
					
					break;
				}
			}
		}
			
		// If we  were not at a vertex, check if we are at a face
		// If so, move a to the boundary (such that it is also contained in neighbor for later intersection search)
		// If b is inside neighbor, we are done
		// If previous segment was an intersection, we already are at correct cell, so do nothing
		if ( (!done) && (!previous_segment_intersected) ) {
			// TODO: This produces the wrong result in some strange situations when face is aligned with one of the coordinate axis
			//       We try to handle this below.
			Point<dim> a_unit = mapQ1.mapping.transform_real_to_unit_cell(cell, *a);
			
			// Check if point is on (or really close to) a face
			int face = -1;
			if (a_unit[0] < tol) {
				face = 0;
				a_unit[0] = 0.0;
			}
			else if (a_unit[0] > 1.0-tol) {
				face = 1;
				a_unit[0] = 1.0;
			}
			else if (a_unit[1] < tol) {
				face = 2;
				a_unit[1] = 0.0;
			}
			else if (a_unit[1] > 1.0-tol) {
				face = 3;
				a_unit[1] = 1.0;
			}
			
			// If point on cell boundary
			if (face > -1) {
				Assert(face<4, ExcInternalError());
				
				// Include extra gard if face is aligned with one of the coordinate axis as 
				// transform_real_to_unit_cell sometimes return the wrong result.
				const Point<dim> v0 = cell->vertex(GeometryInfo<2>::face_to_cell_vertices(face,0));
				const Point<dim> v1 = cell->vertex(GeometryInfo<2>::face_to_cell_vertices(face,1));
				bool really_on_face = true;
				if ( (abs(v0[0] - v1[0]) < tol) && (abs((*a)[0]-v0[0]) > tol) )
					really_on_face = false;
				else if ( (abs(v0[1] - v1[1]) < tol) && (abs((*a)[1]-v0[1]) > tol) )
					really_on_face = false;
				
				if (really_on_face) {
					// Set a equal to point on face and set cell to active neighbor
					*a = mapQ1.mapping.transform_unit_to_real_cell(cell, a_unit);
					Assert(cell->neighbor(face)->point_inside(*a), ExcMessage("a should really be inside neigbor now"));
					typename Triangulation<dim>::active_cell_iterator cell_copy = cell;
					cell = find_active_neighbor_on_face(cell, face, *a);
					
					// Check if point b is in cell
					if (cell->point_inside(*b)) {
						done = true;
						intersections.push_back(Intersection<dim>(vi, cell_copy, cell,
																face, GeometryInfo<dim>::opposite_face[face]));
						
					}
				}
				else {
					std::cout << "Error! Seems like transform_real_to_unit_cell(...) returned wrong result." << std::endl;
					exit(1);
				}
			}
		}
		
		cell->set_material_id(1);
		segment_to_cell.push_back(IndexPair(cell->level(), cell->index()));
		previous_segment_intersected = false;
		
		// If done, proceed to next point
		if (done) {
			++a;
			++vi;
			
			fracture_length_per_cell[cell->active_cell_index()] += a_prev.distance(*b);
		}
		
		// Else, we need to find intersection point
		else {
			Point<dim> x;
			intersection_face = boundary_intersection(x, cell, *a, *b);
			previous_segment_intersected = true;
			
			if (cell->at_boundary(intersection_face))
				break;
			
			const Point<2> v0 = cell->vertex(GeometryInfo<2>::face_to_cell_vertices(intersection_face,0));
			const Point<2> v1 = cell->vertex(GeometryInfo<2>::face_to_cell_vertices(intersection_face,1));
			// If close to a vertex set equal to the vertex
			bool at_vertex = true;
			const typename Triangulation<dim>::active_cell_iterator cell_copy = cell;
			if (x.distance(v0) < tol) {
				x = v0;
				cell = find_active_cell_around_vertex(cell, GeometryInfo<2>::face_to_cell_vertices(intersection_face,0), *b);
			}
			else if (x.distance(v1) < tol) {
				x = v1;
				cell = find_active_cell_around_vertex(cell, GeometryInfo<2>::face_to_cell_vertices(intersection_face,1), *b);
			}
			else
				at_vertex = false;
			
			if (! at_vertex) {
				// If x is in neigbor, update cell and proceed
				typename Triangulation<dim>::cell_iterator neighbor = cell->neighbor(intersection_face);
				if (neighbor->point_inside(x))
					cell = find_active_neighbor_on_face(cell, intersection_face, x);
				// If not, move x a tiny piece along b-a and find new cell
				else {
					for ( unsigned int i=0; i<10; ++i ) {
						if (cell->point_inside(x))
							x = x + (*b-*a)/b->distance(*a) * tol * L/5.0;
						if (neighbor->point_inside(x)) {
							cell = find_active_neighbor_on_face(cell, intersection_face, x);
							done = true;
							break;
						}
					}
					Assert(done, ExcMessage("Not able to find neighbor. Increase tolerance?"));
				}
			}
			
			Assert(a->distance(x) > tol, ExcMessage("Intersection point is equal to first point of segment. Should not end up here. Increase tolerance?"));
			
			++vi;
			if (at_vertex)
				add_vertex_intersection(vi, *a, x, cell_copy, cell);
			else
				intersections.push_back(Intersection<dim>(vi, cell_copy, cell, intersection_face, GeometryInfo<dim>::opposite_face[intersection_face]));
			
			// If b is equal to x, then do not insert x
			if (b->distance(x) > tol)
				a = vertices.insert(b, x);
			else {
				*b = x;
				a = b;
			}
			
			fracture_length_per_cell[cell_copy->active_cell_index()] += a_prev.distance(x);
		}
		
		b = a+1;
	}
	intersection_face = on_cell_boundary(cell, vertices.back());
	if (intersection_face>-1) intersections.push_back(Intersection<dim>(n_vertices()-1, cell, intersection_face));
	
	segment_to_cell.shrink_to_fit();
	Assert(segment_to_cell.size() == n_vertices() -1, ExcInternalError());
	
	return fracture_length_per_cell;
}


template <int dim>
void EmbeddedSurface<dim>::add_vertex_intersection(unsigned int vi, Point<dim> prev, Point<dim> x,
												   typename Triangulation<dim>::active_cell_iterator cell1,
												   typename Triangulation<dim>::active_cell_iterator cell2)
{
	Tensor<1,dim> diff = x - prev;
	
	Assert(diff.norm() > 0, ExcMessage("Points should not be equal."));
	
	unsigned int face_1, face_2;
	if (abs(diff[0]) < abs(diff[1])) {
		face_1 = (diff[1]>0) ? 3 : 2;
		face_2 = (diff[0]>0) ? 1 : 0;
	}
	else {
		face_1 = (diff[0]>0) ? 1 : 0;
		face_2 = (diff[1]>0) ? 3 : 2;
	}
	typename Triangulation<dim>::active_cell_iterator cell_tmp1 = find_active_neighbor_on_face(cell1, face_1, x);
	intersections.push_back(Intersection<dim>(vi, cell1, cell_tmp1,
											  face_1, GeometryInfo<dim>::opposite_face[face_1]));
	if (cell_tmp1->active_cell_index() != cell2->active_cell_index()) {
		typename Triangulation<dim>::active_cell_iterator cell_tmp2 = find_active_neighbor_on_face(cell_tmp1, face_2, x);
		Assert(cell_tmp2->active_cell_index() == cell2->active_cell_index(), ExcInternalError());
		intersections.push_back(Intersection<dim>(vi, cell_tmp1, cell2,
												  face_2, GeometryInfo<dim>::opposite_face[face_2]));
	}
}


template <int dim>
Point<dim> EmbeddedSurface<dim>::vertex(unsigned int i) const
{
	AssertIndexRange(i, n_vertices());
	return vertices[i];
}


template <int dim>
typename EmbeddedSurface<dim>::IndexPair EmbeddedSurface<dim>::get_segment_cell_info(unsigned int i) const
{
	Assert(segment_to_cell.size() == n_vertices()-1, ExcMessage("Wrong size of segment_to_cell. Did you call initialize?"));
	AssertIndexRange(i, n_vertices()-1);
	return segment_to_cell[i];
}


template <int dim>
void EmbeddedSurface<dim>::print_to_screen(const Triangulation<dim>& tria) const
{
	typedef typename DoFHandler<dim>::active_cell_iterator ACI;
	
	std::cout << "Surface information:" << std::endl << std::endl;
	for (unsigned int segment=0; segment<n_vertices()-1; ++segment) {
		IndexPair segment_cell_info = get_segment_cell_info(segment);
		ACI cell = ACI(&tria, segment_cell_info.first, segment_cell_info.second);
		
		std::cout << "Segment #" << segment << ": "
		          << "[" << vertex(segment)   << "] --> "
				  << "[" << vertex(segment+1) << "]" << std::endl;
		std::cout << "  Cell: #" << cell->active_cell_index() 
				  << " (center = [" << cell->center() << "])" << std::endl;
		std::cout << std::endl;
	}
	
	std::cout << "Intersection vertices: " << std::endl;
	for (unsigned int i=0; i<intersections.size(); ++i)
		std:: cout << vertex(intersections[i].index) << std::endl;
	std::cout << std::endl;
}


// Output polygonal chain to vtk file
template <int dim>
void EmbeddedSurface<dim>::output_to_vtk(std::string file_base) const
{
	// Write polygonal chain
	std::ostringstream file_name;
	file_name << "output/" << file_base << ".vtk";
	const std::string file_poly = file_name.str();
	std::ofstream out_poly(file_poly, std::ofstream::out | std::ofstream::trunc);
	out_poly << "# vtk DataFile Version 2.0" << std::endl
			 << "Polygonal chain" << std::endl
			 << "ASCII" << std::endl
			 << "DATASET UNSTRUCTURED_GRID" << std::endl
			 << "POINTS " << n_vertices() << " float" << std::endl;
	for (unsigned int vi=0; vi<n_vertices(); ++vi)
		out_poly << vertex(vi) << " 0" << std::endl;
	out_poly << "CELLS " << n_vertices()+1 << " " << n_vertices()+1 + 2*n_vertices() << std::endl;
	out_poly << n_vertices();
	for (unsigned int i=0; i<n_vertices(); ++i)
		out_poly << " " << i;
	out_poly << std::endl;
	for (unsigned int i=0; i<n_vertices(); ++i)
		out_poly << "1 " << i << std::endl;
	out_poly << "CELL_TYPES " << n_vertices()+1 << std::endl
			 << "4" << std::endl;
	for (unsigned int i=0; i<n_vertices(); ++i)
		out_poly << "1" << std::endl;
	out_poly.close();
}


// Output grid and polygonal chain to vtk file
template <int dim>
void EmbeddedSurface<dim>::output_to_vtk(std::string file_base, const Triangulation<dim>& tria) const
{
	// Write grid
	const std::string file_grid = "output/grid.vtk";
	std::ofstream out_grid(file_grid);
	GridOut grid_out;
	grid_out.write_vtk(tria, out_grid);
	out_grid.close();
	
	// Write polygonal chain
	output_to_vtk(file_base);
}


class FractureNetwork : public Subscriptor
{
public:
	FractureNetwork() : Subscriptor() {}
	FractureNetwork(std::vector<EmbeddedSurface<2>> f) : Subscriptor(), fractures(f) {}
	FractureNetwork(ProblemType pt);
	
	void add(FractureParametrization* parametrization, unsigned int n = 1);
	void add(EmbeddedSurface<2> fracture) { fractures.push_back(fracture); }
	
	template <int dim>
	void init_fractures(const Triangulation<dim>& tria, unsigned int n_qpoints = 2);
	
	template <class MeshType>
	double total_fracture_length_cell(const typename MeshType::active_cell_iterator cell)
	{ return  fracture_length_per_cell[cell->active_cell_index()]; }
	
	
	void output_to_vtk(std::string file_base) const;
	
	typedef typename std::vector<EmbeddedSurface<2>>::const_iterator ConstIterator;
	typedef typename std::vector<EmbeddedSurface<2>>::iterator Iterator;
	
	ConstIterator begin() const { return fractures.begin(); }
	ConstIterator end()   const { return fractures.end();   }
	Iterator begin() { return fractures.begin(); }
	Iterator end()   { return fractures.end();   }
	
	unsigned int n_fractures() const { return fractures.size(); }
	
	template <int dim>
	void print_to_screen(const Triangulation<dim>& tria) const;
	
private:
	std::vector<EmbeddedSurface<2>> fractures;
	std::vector<FractureParametrization*> parametrizations;
	
	Vector<double> fracture_length_per_cell;
	
	void load_fractures(std::string csv_file);
};


FractureNetwork::FractureNetwork(ProblemType pt)
{
	switch (pt)
	{
		case FRACTURE_ANALYTIC:
			add(new FractureAnalytic());
			break;
		case SIMPLE_FRAC:
			add(EmbeddedSurface<2>(Point<2>(1.0/3.0,1.0/3.0), Point<2>(1.0,13.0/16.0)));
			//add(EmbeddedSurface<2>(Point<2>(0.0,0.4), Point<2>(0.4,0.4), 10));
			//add(new FractureArc(Point<2>(1.0,3.0/20.0), 13.0/20.0, acos(-12.0/13.0), PI/2.0), 15);
			//add(new FractureArc(Point<2>(1.0,11.0/10.0), sqrt(85)/10.0, 2*PI-acos(-6.0/sqrt(85)), 2*PI-acos(-2.0/sqrt(85))), 10);
			break;
		case FLEMISCH:
			add(EmbeddedSurface<2>(Point<2>(0.0,0.5), Point<2>(1.0,0.5)));
			add(EmbeddedSurface<2>(Point<2>(0.5,0.0), Point<2>(0.5,1.0)));
			add(EmbeddedSurface<2>(Point<2>(0.5,0.75), Point<2>(1.0,0.75)));
			add(EmbeddedSurface<2>(Point<2>(0.75,0.5), Point<2>(0.75,1.0)));
			add(EmbeddedSurface<2>(Point<2>(0.5,0.625), Point<2>(0.75,0.625)));
			add(EmbeddedSurface<2>(Point<2>(0.625,0.5), Point<2>(0.625,0.75)));
			break;
		case COMPLEX_NETWORK:
			load_fractures("network.csv");
			break;
		case FRACTURE_TEST:
			add(EmbeddedSurface<2>(Point<2>(0.0,0.5), Point<2>(1.0,0.5)));
			break;
		case FRACTURE_INCLINED:
			add(EmbeddedSurface<2>(Point<2>(0.0,1.0/3.0), Point<2>(1.0,2.0/3.0)));
			break;
		case FRACTURE_ANGLE:
			add(EmbeddedSurface<2>(Point<2>(0.0,1.0/5.0), Point<2>(0.5,0.5)));
			add(EmbeddedSurface<2>(Point<2>(0.5,0.5), Point<2>(0.0,4.0/5.0)));
			break;
		default:
			// do nothing
			break;
	}
}


void FractureNetwork::add(FractureParametrization* parametrization, unsigned int n)
{
	parametrizations.push_back(parametrization);
	fractures.push_back(EmbeddedSurface<2>(parametrizations.back(), n));
}


void FractureNetwork::output_to_vtk(std::string file_base) const
{
	unsigned int i = 0;
	for (ConstIterator it=begin(); it!=end(); ++it, ++i) {
		std::string file_name = file_base;
		file_name += "-";
		file_name += std::to_string(i);
		it->output_to_vtk(file_name);
	}
}


template <>
void FractureNetwork::init_fractures(const Triangulation<3>&, unsigned int)
{
	ExcNotImplemented();
}


template <>
void FractureNetwork::init_fractures(const Triangulation<2>& tria, unsigned int n_qpoints)
{
	// Cells containing fracture will be given material id 1
	// Set material id to 0 for all cells here
	{
		typename Triangulation<2>::cell_iterator
		cell = tria.begin(0),
		endc = tria.end(0);
		for ( ; cell != endc; ++cell)
			cell->recursively_set_material_id(0);
	}
	
	// TODO: Check that we get correct fracture_length
	fracture_length_per_cell.reinit(tria.n_active_cells());
	for (Iterator f = fractures.begin(); f != fractures.end(); ++f) {
		f->setup(tria, n_qpoints);
		fracture_length_per_cell += f->initialize(tria);
	}
}


void FractureNetwork::load_fractures(std::string csv_file)
{
	std::cout << "Reading fractures from file..." << std::endl;
	std::ifstream filestream;
	filestream.open(csv_file);
	if (! filestream.good()) {
		std::cout << "Error reading fracture file " << csv_file << std::endl;
		exit(1);
	}
	unsigned int fid;
	unsigned int count = 0;
	Point<2> p1;
	Point<2> p2;
	std::string next_line;
	char tmp;
	// Skip first line
	std::getline(filestream, next_line);
	while (std::getline(filestream, next_line)) {
		std::stringstream line(next_line, std::ios_base::in);
		line >> fid;
		line.get(tmp);
		line >> p1[0];
		line.get(tmp);
		line >> p1[1];
		line.get(tmp);
		line >> p2[0];
		line.get(tmp);
		line >> p2[1];
		add(EmbeddedSurface<2>(p1, p2));
		++count;
	}
	std::cout << "Read and loaded " << count << " fractures from " << csv_file << std::endl;
	
	filestream.close();
}


template <>
void FractureNetwork::print_to_screen(const Triangulation<3>&) const
{
	ExcNotImplemented();
}

template <>
void FractureNetwork::print_to_screen(const Triangulation<2>& tria) const
{
	unsigned int fid =0;
	for (ConstIterator f = fractures.begin(); f != fractures.end(); ++f, ++fid) {
		std::cout << "FRACTURE " << fid << std::endl;
		f->print_to_screen(tria);
	}
}



// Utility functions


// Compute unit normal to tensor
template <int dim>
Tensor<1,dim> unit_normal(Tensor<1,dim>)
{
	Assert(false, ExcNotImplemented());
	return Tensor<1,dim>();
}

Tensor<1,2> unit_normal(Tensor<1,2> v)
{
	Assert(v.norm() > 1e-15, ExcMessage("Can't find normal vector of zero."))
	
	Tensor<1,2> n;
	n[0] =  v[1];
	n[1] = -v[0];
	n = n/n.norm();
	
	Assert(abs(n.norm()-1.0) < 1e-15, ExcMessage("Normal vector not of unit size"));
	Assert(abs(v*n) < 1e-15*v.norm(), ExcMessage("Vector is not normal to input vector"));
	
	return n;
}


// Compute unit tangent
template <int dim>
Tensor<1,dim> unit_tangent(Point<dim> a, Point<dim>b)
{
	return (b-a)/b.distance(a);
}


// Projection matrix (I - n*n^T) for line segment a->b
template <int dim>
Tensor<2,dim> projection_matrix(Point<dim> a, Point<dim> b)
{
	Tensor<2,dim> P;
	// Set P=I
	for (unsigned int i=0; i<dim; ++i)
		P[i][i] = 1.0;
	Tensor<1,dim> n = unit_normal(b-a);
	P = P - outer_product(n,n);
	Tensor<2,dim> P2 = P*P - P;
	Assert(P2.norm() < 1e-15, ExcMessage("P^2 != P"));
	return P;
}


// Find intersection point s of lines a-b and c-d
bool line_intersection(Point<2> a, Point<2> b, Point<2> c, Point<2> d, Point<2>& s)
{
	FullMatrix<double> M(2);
	Vector<double> r(2);
	M.set(0,0, a[1]-b[1]);
	M.set(0,1, b[0]-a[0]);
	M.set(1,0, c[1]-d[1]);
	M.set(1,1, d[0]-c[0]);
	r[0] = (b[0]-a[0])*a[1] - (b[1]-a[1])*a[0];
	r[1] = (d[0]-c[0])*c[1] - (d[1]-c[1])*c[0];
	// Try to invert
	M.invert(M);
	Vector<double> s_vec(2);
	s_vec[0] = s[0]; s_vec[1] = s[1];
	M.vmult(s_vec,r);
	s = Point<2>(s_vec[0], s_vec[1]);
	if (std::isnan(s[0]) || std::isnan(s[0]))
		return false;
	else
		return true;
}


// Find cell boundary intersection
template <int dim>
unsigned int boundary_intersection(Point<dim>&, typename Triangulation<dim>::active_cell_iterator, Point<dim>, Point<dim>)
{
	Assert(false, ExcNotImplemented());
	return 0;
}

template <>
unsigned int boundary_intersection<2>(Point<2>& x, typename Triangulation<2>::active_cell_iterator cell, Point<2> a, Point<2> b)
{
	//const double tol = 1e-15;

	Assert(cell->point_inside(a),  ExcMessage("Point a is not inside cell"));
	Assert(!cell->point_inside(b), ExcMessage("Point b *is* inside cell"));
	
	static const int direction[4] = {1, -1, -1, 1};
	
	// Loop through faces
	bool found_intersection = false;
	for (unsigned int face=0; face<GeometryInfo<2>::faces_per_cell; ++face) {
		const Point<2> v0 = cell->vertex(GeometryInfo<2>::face_to_cell_vertices(face,0));
		const Point<2> v1 = cell->vertex(GeometryInfo<2>::face_to_cell_vertices(face,1));
		// Check if (b-a)*n < 0. Then this face is in the wrong direction
		if (direction[face] * unit_normal(v1-v0) * (b-a) > 0)
			continue;
		found_intersection = line_intersection(a,b,v0,v1, x);
		double face_length = cell->face(face)->measure();
		// Check if computed intersection is on element face
		if (v0.distance(x) > face_length || v1.distance(x) > face_length)
			found_intersection = false;
		if (found_intersection)
			return face;
	}
	std::cout << "Error! Unable to find intersection with cell boundary!" << std::endl
			  << "       Cell center: " << cell->center() << std::endl
			  << "       Points:      " << a << "  &  " << b << std::endl;
	exit(1);
}


// Find active neighbor to 'cell' on local face 'face'
template <int dim>
typename Triangulation<dim>::active_cell_iterator find_active_neighbor_on_face(typename Triangulation<dim>::active_cell_iterator cell, 
																			   const unsigned int face, const Point<dim> x)
{
	// If at boundary, do nothing
	if (cell->at_boundary(face))
		return cell;
	typename Triangulation<dim>::cell_iterator neighbor = cell->neighbor(face);
	if (neighbor->active())
		return neighbor;
	else {
		for (unsigned int subface=0; subface<GeometryInfo<dim>::max_children_per_face; ++subface) {
			neighbor = cell->neighbor_child_on_subface(face, subface);
			if (neighbor->point_inside(x))
				return neighbor;
		}
		// If we are here, then something went wrong.
		Assert(false, ExcMessage("Not able to find child containing the point x. Can be due to boundary_intersection(...) not returning a point exactly on the boundary"))
	}
	return cell;
}


// Return active cell containing x that shares the (local) vertex 'vertex_i' of 'cell'
template <int dim>
typename Triangulation<dim>::active_cell_iterator find_active_cell_around_vertex(typename Triangulation<dim>::active_cell_iterator cell, 
																				 const unsigned int vertex_i, const Point<dim> x)
{
	std::vector<typename Triangulation<dim>::active_cell_iterator> possible_cells = 
	GridTools::find_cells_adjacent_to_vertex(cell->get_triangulation(), cell->vertex_index(vertex_i));
	for (unsigned int ci = 0; ci<possible_cells.size(); ++ci) {
		if (possible_cells[ci]->point_inside(x))
			return possible_cells[ci];
	}
	// If we are here, something went wrong, or distance between nodes are larger than h_min
	Assert(false, ExcMessage("Could not find active cell around vertex. Most probably because distance between nodes are larger than smallest face length?"));
	return cell;
}


template <int dim>
int on_cell_boundary(const typename Triangulation<dim>::active_cell_iterator cell, const Point<dim> p)
{
	const double tol = 1e-12;
	
	int face = -1;
	
	StaticMappingQ1<dim> mapQ1;
	Point<dim> p_unit = mapQ1.mapping.transform_real_to_unit_cell(cell, p);
	
	for (unsigned int d=0; d<dim; ++d) {
		if (p_unit[d] < tol) {
			face = 2*d;
			break;
		}
		else if ((1.0 - p_unit[d]) < tol) {
			face = 2*d+1;
			break;
		}
	}
	return face;
}


#endif /* EMBEDDED_SURFACE_H */
