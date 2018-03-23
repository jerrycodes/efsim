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

#ifndef ELLIPTIC_PRESSURE_SOLVER_H
#define ELLIPTIC_PRESSURE_SOLVER_H

#include "PressureSolverBase.h"
#include "EmbeddedSurface.h"
#include <deal.II/base/index_set.h>
#include <deal.II/grid/filtered_iterator.h>


using namespace dealii;


// Solver for elliptic (stationary) pressure equation
template <int dim>
class EllipticPressureSolver : public PressureSolverBase<dim>
{
public:
	// Constructor calls base constructor
	EllipticPressureSolver(Triangulation<dim> &tria, const FE_Q<dim> &fe, DoFHandler<dim> &dh,
						   const RockProperties<dim> &r, ProblemFunctionsFlow<dim> &f)
		: PressureSolverBase<dim>(tria, fe, dh, r, f), fracture_network(new FractureNetwork())
		{}

	~EllipticPressureSolver() {}

	void run();
	void run_dual();
	
	void set_fractures(FractureNetwork& frac, double perm, double width);
	void set_stabilization(double sigma) { stabilize = true; sigma_stab = sigma;}
	
	// Dummy function that do nothing. Only here to comply with ParabolicPressureSolver
	void solve_time_step(double) {}
	
	void add_fracture_velocity(VelocityData<dim>& velocity, double frac_width) const;
	void output_fracture_pressure() const;

	void print_timing();

private:

	std::string filename() const;
	
	void create_sparsity_pattern();
	void assemble_dual_rhs();
	void output_dual_results() const;
	void assemble_surface_contrib();
	void add_cell_contrib_surface_segment(const typename DoFHandler<dim>::active_cell_iterator cell, 
										  const Point<dim> a, const Point<dim> b);
	void add_neumann_contrib_fracture(const EmbeddedSurface<dim> fracture);
	void stabilize_fractures();
	void condense_contribution(const std::vector<unsigned int> dofs, const FullMatrix<double> matrix,
										std::vector<unsigned int>& dofs_red, FullMatrix<double>& matrix_red);
	void assemble_stabilization_face_contribution(const FEFaceValuesBase<dim>& fe_face_values_cell,
												  const FEFaceValuesBase<dim>& fe_face_values_neighbor,
												  FullMatrix<double>& local_matrix);
	void output_system_matrix() const;
	
	SmartPointer<FractureNetwork> fracture_network;
	double fracture_perm;
	double fracture_width;
	
	// Timer variables
	Timer t_assembly;
	Timer t_solve;
	
	bool stabilize = false;
	double sigma_stab;
	LA::Matrix stabilization_matrix;
	
	unsigned int cycle_count = 0;
};


/* Main driver function
 * Setup, assemble and solve linear system. Then output results.
 */
template <int dim>
void EllipticPressureSolver<dim>::run()
{
	++cycle_count;
	
	this->setup_system();
	
	t_assembly.start ();
	this->assemble_laplace();
	this->assemble_rhs();
	assemble_surface_contrib();
	this->system_matrix.copy_from(this->laplace_matrix);
	if (! this->weak_bcs) this->apply_dirichlet_strongly();
	t_assembly.stop ();
	
	// If pure Neumann problem, spesify first dof to coincide with Dirichlet condition
	// To this by double A_ii and then b_i += A_ii*p_B
	if (this->pure_neumann) {
		const int dof_no = 0;
		const Point<dim> dof_pos = this->dof_handler->begin_active()->vertex(dof_no);
		const double entry = this->system_matrix(dof_no, dof_no);
		this->system_matrix.add(dof_no, dof_no, entry);
		this->system_rhs(dof_no) += entry*this->funs->dirichlet->value(dof_pos);
	}

	t_solve.start ();
	this->solve_linsys();
	t_solve.stop ();
	
	output_system_matrix();
	this->output_results();
}


template <int dim>
void EllipticPressureSolver<dim>::run_dual()
{
	// Use same system matrix as for primal problem
	
	reinit_vec_seq(this->dual_sol, this->dof_handler->n_dofs());
	assemble_dual_rhs();
	if (! this->weak_bcs) this->apply_dirichlet_strongly(true);
	
	this->solve_linsys(true);
	
	this->output_dual_results();
}


template <int dim>
void EllipticPressureSolver<dim>::set_fractures(FractureNetwork& frac, double perm, double width)
{
	Assert(dim == 2 || frac.n_fractures() == 0, ExcNotImplemented());
	fracture_network = &frac;
	fracture_perm = perm;
	fracture_width = width;
}


template <int dim>
void EllipticPressureSolver<dim>::create_sparsity_pattern()
{
	DynamicSparsityPattern c_sparsity(this->dof_handler->n_dofs());
	DoFTools::make_sparsity_pattern(*(this->dof_handler), c_sparsity, this->constraints, false);
	
	if (stabilize) {
		
		const unsigned int   dofs_per_cell = this->fe->dofs_per_cell;
		std::vector<unsigned int> cell_dofs(dofs_per_cell);
		std::vector<unsigned int> neighbor_dofs(dofs_per_cell);
		
		// Loop over fractured cells and their neighbors
		FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
		cell (IteratorFilters::MaterialIdEqualTo(1), this->dof_handler->begin_active()),
		endc (IteratorFilters::MaterialIdEqualTo(1), this->dof_handler->end());
		for ( ; cell != endc; ++cell) {
			cell->get_dof_indices(cell_dofs);
			
			std::vector<typename DoFHandler<dim>::active_cell_iterator> neighbors;
			GridTools::get_active_neighbors<DoFHandler<dim>>(cell, neighbors);
			
			for (unsigned int ni=0; ni<neighbors.size(); ++ni) {
				neighbors[ni]->get_dof_indices(neighbor_dofs);
				for (unsigned int i=0; i<dofs_per_cell; ++i)
					c_sparsity.add_entries(cell_dofs[i], neighbor_dofs.begin(), neighbor_dofs.end());
			}
		}
		c_sparsity.symmetrize();
		
	}
	
	this->sparsity_pattern.copy_from(c_sparsity);
}



template <int dim>
void EllipticPressureSolver<dim>::assemble_dual_rhs()
{
	Assert(! this->weak_bcs, ExcNotImplemented());
	
	this->system_rhs = 0;
	
	// TODO: Take this as input. Put it to one here, to get something else than the trivial solution.
	Tensor<1,dim> one;
	for (int d=0; d<dim; ++d) one[d] = 1.0;
	std::vector<Tensor<1,dim>> dual_rhs_values(this->triangulation->n_active_cells(), one);
	
	QGauss<dim>   quadrature_formula(this->n_qpoints);
	FEValues<dim> fe_values (*(this->fe), quadrature_formula,
							 update_gradients | update_JxW_values);
	
	const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
	const unsigned int n_q_points    = quadrature_formula.size();
	Vector<double> cell_rhs (dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	
	// Main loop
	typename DoFHandler<dim>::active_cell_iterator
	cell = this->dof_handler->begin_active(),
	endc = this->dof_handler->end();
	for (; cell!=endc; ++cell) {
		cell_rhs = 0;
		fe_values.reinit(cell);
		const Tensor<2,dim> permeability_cell = this->rock->get_perm(cell);
		const Tensor<1,dim> dual_rhs_value = dual_rhs_values[cell->active_cell_index()];
		
		for (unsigned int q=0; q<n_q_points; ++q) {
			const double JxW = fe_values.JxW(q);
			for (unsigned int i = 0; i<dofs_per_cell; i++) {
				cell_rhs(i) += (permeability_cell * fe_values.shape_grad(i,q)) * dual_rhs_value * JxW;
			}
		}
		cell->get_dof_indices (local_dof_indices);
		this->constraints.distribute_local_to_global(cell_rhs, local_dof_indices, this->system_rhs);
	}
}


template <>
void EllipticPressureSolver<3>::assemble_surface_contrib()
{
	Assert(fracture_network->n_fractures() == 0, ExcNotImplemented());
}


// Assemble surface contribution along fractures.
template <>
void EllipticPressureSolver<2>::assemble_surface_contrib()
{
	typedef typename DoFHandler<2>::active_cell_iterator ACI;
	//const double delta = 1e-15;
	
	// Loop over fractures
	for (FractureNetwork::ConstIterator fracture = fracture_network->begin();
		 fracture != fracture_network->end(); ++fracture) {
		// Iterate through vertices defining fracture
		for (unsigned int vi=0; vi<fracture->n_vertices()-1; ++vi) {
			const Point<2> a = fracture->vertex(vi);
			const Point<2> b = fracture->vertex(vi+1);
			typename EmbeddedSurface<2>::IndexPair cell_info = fracture->get_segment_cell_info(vi);
			ACI cell = ACI(this->triangulation, cell_info.first, cell_info.second, this->dof_handler);
			add_cell_contrib_surface_segment(cell, a, b);
		}
		
		// Add contribution on Neumann boundary
		add_neumann_contrib_fracture(*fracture);
	}
	
	if (stabilize) {
		stabilization_matrix.reinit(this->sparsity_pattern);
		stabilize_fractures();
		this->laplace_matrix.add(1.0, stabilization_matrix);
		
		// Test
		/*
		LA::Vec linear_function, should_be_zero;
		reinit_vec_seq(linear_function, this->dof_handler->n_dofs());
		reinit_vec_seq(should_be_zero, this->dof_handler->n_dofs());
		
		VectorTools::interpolate(*(this->dof_handler), LinearDecreasing<2>(0), linear_function);
		stabilization_matrix.vmult(should_be_zero, linear_function);
		std::cout << "L2 norm of stab matrix times linear function: " << should_be_zero.l2_norm() << std::endl;
		*/
	}
}


template <int dim>
void EllipticPressureSolver<dim>::add_cell_contrib_surface_segment(const typename DoFHandler<dim>::active_cell_iterator cell, 
																   const Point<dim> a, const Point<dim> b)
{
	StaticMappingQ1<dim> mapQ1;
	
	const Point<dim> midpoint = (a+b)/2.0;
	const Quadrature<dim> quadrature(mapQ1.mapping.transform_real_to_unit_cell(cell, midpoint));
	const unsigned int q_point = 0;
	
	FEValues<dim> fe_values(*(this->fe), quadrature, update_values | update_gradients);
	fe_values.reinit(cell);
	
	const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
	
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	cell->get_dof_indices(local_dof_indices);
	
	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double>     cell_rhs(dofs_per_cell);
	
	const Tensor<2,dim> P = projection_matrix(a,b);
	const double rhs_value = this->funs->right_hand_side_fracture->value(midpoint,0);
	const double segment_length = a.distance(b);
	
	for (unsigned int i=0; i<dofs_per_cell; ++i) {
		for (unsigned int j=0; j<dofs_per_cell; ++j) {
			cell_matrix(i,j) = fe_values.shape_grad(j,q_point) * P * fe_values.shape_grad(i,q_point);
		}
		cell_rhs(i) = fe_values.shape_value(i,q_point);
	}
	cell_matrix *= segment_length * fracture_width * fracture_perm;
	cell_rhs *= segment_length * rhs_value;
	this->constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, this->laplace_matrix, this->system_rhs);
}


template <int dim>
void EllipticPressureSolver<dim>::add_neumann_contrib_fracture(const EmbeddedSurface<dim> fracture)
{
	typedef typename DoFHandler<2>::active_cell_iterator ACI;
	StaticMappingQ1<dim> mapQ1;
	const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
	
	std::vector<Intersection<dim>> end_intersections;
	end_intersections.push_back(fracture.first_intersection());
	end_intersections.push_back(fracture.last_intersection());
	
	typename std::vector<Intersection<dim>>::const_iterator intersection = end_intersections.begin();
	for ( ; intersection != end_intersections.end(); ++intersection) {
		if (intersection->n_cells == 1) {
			const ACI cell(this->triangulation, intersection->cells[0].first, intersection->cells[0].second, this->dof_handler);
			const unsigned int face = intersection->faces[0];
			
			Assert(cell->at_boundary(face), ExcInternalError());
			if (cell->face(face)->boundary_id() == 1) {
				const Point<2> x = fracture.vertex(intersection->index);
				const double neumann_value = this->funs->neumann->value(x);
				
				std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
				cell->get_dof_indices(local_dof_indices);
				
				const Quadrature<dim> quadrature(mapQ1.mapping.transform_real_to_unit_cell(cell, x));
				FEValues<2> fe_values(*(this->fe), quadrature, update_values);
				fe_values.reinit(cell);
				
				for (unsigned int i=0; i<dofs_per_cell; ++i)
					this->constraints.distribute_local_to_global(local_dof_indices[i],
																- fracture_width * neumann_value * fe_values.shape_value(i,0),
																this->system_rhs);
			}
		}
	}
}



template <int dim>
void EllipticPressureSolver<dim>::stabilize_fractures()
{
	typedef typename DoFHandler<2>::active_cell_iterator ACI;
	
	QGauss<dim-1> quad_face(this->n_qpoints);
	
	FEFaceValues<dim> fe_face_values(*(this->fe), quad_face,
									 update_gradients | update_normal_vectors | update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> fe_face_values_neighbor(*(this->fe), quad_face,
											  update_gradients);
		
	const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
	const unsigned int dofs_per_interface = 2*dofs_per_cell;
	
	std::vector<unsigned int> cell_dofs(dofs_per_cell);
	std::vector<unsigned int> neighbor_dofs(dofs_per_cell);
	std::vector<unsigned int> interface_dofs(dofs_per_interface);
	FullMatrix<double> interface_matrix(dofs_per_interface);
		
	this->triangulation->clear_user_flags();
	
	// Loop over fractured cells
	FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
	cell (IteratorFilters::MaterialIdEqualTo(1), this->dof_handler->begin_active()),
	endc (IteratorFilters::MaterialIdEqualTo(1), this->dof_handler->end());
	for ( ; cell != endc; ++cell) {
			
			cell->get_dof_indices(cell_dofs);
			for (unsigned int i=0; i<dofs_per_cell; ++i)
				interface_dofs[i] = cell_dofs[i];
			
			for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
				
				interface_matrix = 0;
				
				// Do nothing on boundary faces
				if (cell->at_boundary(face))
					continue;
				
				Assert(cell->neighbor_level(face) == cell->level(), ExcMessage("Assumes equally refined neighbors along fracture."));
				const ACI neighbor = cell->neighbor(face);
				
				// Skip faces already visited
				if (neighbor->user_flag_set())
					continue;
				
				neighbor->get_dof_indices(neighbor_dofs);
				for (unsigned int i=0; i<dofs_per_cell; ++i)
					interface_dofs[i+dofs_per_cell] = neighbor_dofs[i];
				
				fe_face_values.reinit(cell, face);
				fe_face_values_neighbor.reinit(neighbor, GeometryInfo<dim>::opposite_face[face]);
				assemble_stabilization_face_contribution(fe_face_values, fe_face_values_neighbor, interface_matrix);
				interface_matrix *= sigma_stab * cell->face(face)->measure();
				
				std::vector<unsigned int> interface_dofs_red;
				FullMatrix<double> interface_matrix_red;
				condense_contribution(interface_dofs, interface_matrix, interface_dofs_red, interface_matrix_red);
				this->constraints.distribute_local_to_global(interface_matrix_red, interface_dofs_red, stabilization_matrix);
			}
			
			// Mark cell as visited
			cell->set_user_flag();
	}
	
	this->triangulation->clear_user_flags();
}


template <int dim>
void EllipticPressureSolver<dim>::assemble_stabilization_face_contribution(const FEFaceValuesBase<dim>& fe_face_values_cell,
																		   const FEFaceValuesBase<dim>& fe_face_values_neighbor,
																		   FullMatrix<double>& local_matrix)
{
	const unsigned int dofs_per_cell = fe_face_values_cell.dofs_per_cell;
	const unsigned int n_q_face_points = fe_face_values_cell.n_quadrature_points;
	
	Tensor<1,dim> unit_normal = fe_face_values_cell.normal_vector(0);
	// Loop over cell dofs
	for (unsigned int i=0; i<dofs_per_cell; ++i) {
		// Connect to cell dofs
		for (unsigned int j=0; j<dofs_per_cell; ++j) {
			double contribution = 0;
			for (unsigned int q_point=0; q_point<n_q_face_points; ++q_point) {
				contribution += (fe_face_values_cell.shape_grad(i, q_point) * unit_normal) *
								(fe_face_values_cell.shape_grad(j, q_point) * unit_normal) *
								fe_face_values_cell.JxW(q_point);
			}
			local_matrix(i,j) += contribution;
		}
		// Connect to neighbor dofs
		for (unsigned int j=0; j<dofs_per_cell; ++j) {
			double contribution = 0;
			for (unsigned int q_point=0; q_point<n_q_face_points; ++q_point) {
				contribution += (fe_face_values_cell.shape_grad(i, q_point) * unit_normal) *
								(fe_face_values_neighbor.shape_grad(j, q_point) * (-1.0) * unit_normal) *
								fe_face_values_cell.JxW(q_point);
			}
			local_matrix(i,j+dofs_per_cell) += contribution;
		}
	}
	
	// Repeat loop for neighbor dofs.
	for (unsigned int i=0; i<dofs_per_cell; ++i) {
		// Connect to own dofs
		for (unsigned int j=0; j<dofs_per_cell; ++j) {
		double contribution = 0;
			for (unsigned int q_point=0; q_point<n_q_face_points; ++q_point) {
				contribution += (fe_face_values_neighbor.shape_grad(i, q_point) * (-1.0) * unit_normal) *
								(fe_face_values_neighbor.shape_grad(j, q_point) * (-1.0) * unit_normal) *
								fe_face_values_cell.JxW(q_point);
			}
			local_matrix(i+dofs_per_cell,j+dofs_per_cell) += contribution;
		}
		// Connect to cell dofs
		for (unsigned int j=0; j<dofs_per_cell; ++j) {
			double contribution = 0;
			for (unsigned int q_point=0; q_point<n_q_face_points; ++q_point) {
				contribution += (fe_face_values_neighbor.shape_grad(i, q_point) * (-1.0) * unit_normal) *
								(fe_face_values_cell.shape_grad(j, q_point) * unit_normal) *
								fe_face_values_cell.JxW(q_point);
			}
			local_matrix(i+dofs_per_cell,j) += contribution;
		}
	}
} 


// Find dofs occuring twice and add up contribution (distribute_local_to_global(...) does not allow the same dof appear twice
template <int dim>
void EllipticPressureSolver<dim>::condense_contribution(const std::vector<unsigned int> dofs, const FullMatrix<double> matrix,
														std::vector<unsigned int>& dofs_red, FullMatrix<double>& matrix_red)
{
	const unsigned int n = dofs.size();
	AssertDimension(n, matrix.n());
	AssertDimension(n, matrix.m());
	
	// Build mapping
	std::vector<unsigned int> mapping(n);
	for (unsigned int i=0; i<n; ++i)
		mapping[i] = i;
	
	unsigned int count_double_dof = 0;
	for (unsigned int i=1; i<n; ++i) {
		for (unsigned int j=0; j<i; ++j) {
			if (dofs[i] == dofs[j]) {
				mapping[i] = mapping[j];
				for (unsigned int k=i+1; k<n; ++k)
					mapping[k] = mapping[k] -1;
				// Set j=n so that second for lopp terminates
				++count_double_dof;
				j=i;
			}
		}
	}
	
	// Apply mapping
	if (count_double_dof > 0) {
		dofs_red.resize(n-count_double_dof);
		matrix_red = FullMatrix<double>(n-count_double_dof);
		for (unsigned int i=0; i<n; ++i) {
			for (unsigned int j=0; j<n; ++j) {
				matrix_red[mapping[i]][mapping[j]] += matrix[i][j];
			}
			dofs_red[mapping[i]] = dofs[i];
		}
	}
	else {
		dofs_red = dofs;
		matrix_red = matrix;
	}
}


template <>
void EllipticPressureSolver<3>::add_fracture_velocity(VelocityData<3>&, double) const
{
	Assert(false, ExcNotImplemented());
}


template <int dim>
void EllipticPressureSolver<dim>::add_fracture_velocity(VelocityData<dim>& velocity, double frac_width) const
{
	typedef typename DoFHandler<dim>::active_cell_iterator ACI;
	typedef typename Intersection<dim>::IndexPair IP;
	
	StaticMappingQ1<dim> mapQ1;
	typename VelocityData<dim>::FE_Pointer fe_velocity = velocity.get_fe();
	typename VelocityData<dim>::DH_Pointer dh_velocity = velocity.get_dh();
	
	std::vector<unsigned int> face_dofs(fe_velocity->n_dofs_per_cell());
	std::vector<Tensor<1,dim>> pressure_gradients(1);
	
	for (FractureNetwork::ConstIterator fracture = fracture_network->begin(); fracture != fracture_network->end(); ++fracture) {
		
		typename std::vector<Intersection<dim>>::const_iterator intersection = fracture->begin_intersection();
		for ( ; intersection != fracture->end_intersection(); ++intersection) {
			const Point<dim> x = fracture->vertex(intersection->index);
			// If at end of fracture
			if (intersection->n_cells == 1) {
				const ACI cell_i = ACI(this->triangulation, intersection->cells[0].first, intersection->cells[0].second, this->dof_handler);
				const unsigned int face_i = intersection->faces[0];
				
				Point<dim> x_loc_i = mapQ1.mapping.transform_real_to_unit_cell(cell_i, x);
				
				Point<dim-1> q_point_i;
				if (face_i == 0 || face_i == 1) {
					q_point_i[0] = x_loc_i[1];
				}
				else {
					Assert(face_i == 2 || face_i == 3, ExcInternalError());
					q_point_i[0] = x_loc_i[0];
				}
				
				FEFaceValues<dim> fe_faces_values_i(*(this->fe), Quadrature<dim-1>(q_point_i), update_gradients);
				fe_faces_values_i.reinit(cell_i, face_i);
				
				Point<dim> a;
				if (intersection->index == 0)
					a = fracture->vertex(1);
				else if (intersection->index == fracture->n_vertices()-1)
					a = fracture->vertex(fracture->n_vertices()-2);
				else
					Assert(false, ExcMessage("If only one cell given, then intersection should be an endpoint."));
				
				fe_faces_values_i.get_function_gradients(this->pressure_sol, pressure_gradients);
				// Tangent points out of cell
				Tensor<1,dim> tangent = unit_tangent(a, x);
				Tensor<1,dim> pressure_grad_gamma = projection_matrix(x, a) * pressure_gradients[0];
				
				const double flux_out = - fracture_width * fracture_perm * tangent * pressure_grad_gamma;
				
				ACI cell_velocity = ACI(this->triangulation, cell_i->level(), cell_i->index(), dh_velocity);
				cell_velocity->get_dof_indices(face_dofs);
				
				// No contribution on Neumann boundary
				if (cell_i->at_boundary(face_i) && cell_i->face(face_i)->boundary_id() == 1) {
					velocity.set_dof_value(face_dofs[face_i], velocity.get_dof_value(face_dofs[face_i]) * frac_width / cell_velocity->face(face_i)->measure());
				}
				else {
					velocity.set_dof_value(face_dofs[face_i], GeometryInfo<dim>::unit_normal_orientation[face_i] * flux_out);
				}
				velocity.add_fracture_intersection_dof(face_dofs[face_i]);
			}
			else if (intersection->n_cells == 2) {
				const ACI cell_i = ACI(this->triangulation, intersection->cells[0].first, intersection->cells[0].second, this->dof_handler);
				const ACI cell_j = ACI(this->triangulation, intersection->cells[1].first, intersection->cells[1].second, this->dof_handler);
				const unsigned int face_i = intersection->faces[0];
				const unsigned int face_j = intersection->faces[1];
				
				Point<dim> x_loc_i = mapQ1.mapping.transform_real_to_unit_cell(cell_i, x);
				Point<dim> x_loc_j = mapQ1.mapping.transform_real_to_unit_cell(cell_j, x);
				
				Point<dim-1> q_point_i, q_point_j;
				if (face_i == 0 || face_i == 1) {
					Assert(face_j == 0 || face_j == 1, ExcInternalError());
					q_point_i[0] = x_loc_i[1];
					q_point_j[0] = x_loc_j[1];
				}
				else {
					Assert(face_i == 2 || face_i == 3, ExcInternalError());
					Assert(face_j == 2 || face_j == 3, ExcInternalError());
					q_point_i[0] = x_loc_i[0];
					q_point_j[0] = x_loc_j[0];
				}
				
				FEFaceValues<dim> fe_faces_values_i(*(this->fe), Quadrature<dim-1>(q_point_i), update_gradients);
				FEFaceValues<dim> fe_faces_values_j(*(this->fe), Quadrature<dim-1>(q_point_j), update_gradients);
				
				fe_faces_values_i.reinit(cell_i, face_i);
				fe_faces_values_j.reinit(cell_j, face_j);
				
				double flux_out = 0.0;
				const Point<dim> a = fracture->vertex(intersection->index-1);
				const Point<dim> b = fracture->vertex(intersection->index+1);
				
				fe_faces_values_i.get_function_gradients(this->pressure_sol, pressure_gradients);
				// Tangent out of cell_i
				Tensor<1,dim> tangent = unit_tangent(a, x);
				Tensor<1,dim> pressure_grad_gamma = projection_matrix(a, x) * pressure_gradients[0];
				
				flux_out -= 0.5 * fracture_width * fracture_perm * tangent * pressure_grad_gamma;
				
				fe_faces_values_j.get_function_gradients(this->pressure_sol, pressure_gradients);
				// Tangent out of cell_i
				tangent = unit_tangent(x, b);
				pressure_grad_gamma = projection_matrix(x, a) * pressure_gradients[0];
				
				flux_out -= 0.5 * fracture_width * fracture_perm * tangent * pressure_grad_gamma;
				
				// Add contribution to finest cell (velocity constraints will handle the other)
				IP cell_info_velocity;
				unsigned int face_velocity;
				if (cell_i->level() >= cell_j->level()) {
					cell_info_velocity = IP(cell_i->level(), cell_i->index());
					face_velocity = face_i;
				}
				else {
					cell_info_velocity = IP(cell_j->level(), cell_j->index());
					face_velocity = face_j;
					// Correct for tangent ouf of cell_i
					flux_out *= -1;
				}
				
				ACI cell_velocity = ACI(this->triangulation, cell_info_velocity.first, cell_info_velocity.second, dh_velocity);
				cell_velocity->get_dof_indices(face_dofs);
				velocity.set_dof_value(face_dofs[face_velocity], GeometryInfo<dim>::unit_normal_orientation[face_velocity] * flux_out);
				velocity.add_fracture_intersection_dof(face_dofs[face_velocity]);
			}
			else {
				Assert(false, ExcNotImplemented());
			}
		}
	}
	velocity.apply_constraints();
}


template <int dim>
std::string EllipticPressureSolver<dim>::filename() const 
{
	std::ostringstream name;
	name << "output/" << this->filename_base << "-" << cycle_count << ".vtk";
	return name.str();
}


template <int dim>
void EllipticPressureSolver<dim>::print_timing()
{
	std::cout << "Timing elliptic pressure solver (wall time in sec):" << std::endl
		  << "  Assembly:     " << t_assembly.wall_time() << std::endl
		  << "  Solve system: " << t_solve.wall_time() << std::endl
		  << "  Sum:          " << t_assembly.wall_time() + t_solve.wall_time() << std::endl;
}


template <int dim>
void EllipticPressureSolver<dim>::output_dual_results() const
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler (*(this->dof_handler));
	data_out.add_data_vector (this->dual_sol, "DualPressure");
	data_out.build_patches ();
	
	std::ostringstream filename;
	filename << "output/" << this->filename_base << "-dual-" << cycle_count << ".vtk";
	std::ofstream output(filename.str().c_str());
	data_out.write_vtk(output);
	
	data_out.clear();
}


template <>
void EllipticPressureSolver<3>::output_fracture_pressure() const
{
	Assert(false, ExcNotImplemented());
}


template <>
void EllipticPressureSolver<2>::output_fracture_pressure() const
{
	const unsigned int dim = 2;
	typedef typename DoFHandler<dim>::active_cell_iterator ACI;
	StaticMappingQ1<dim> mapQ1;
	std::vector<double> pressure_val(1);
	
	unsigned int fid = 1;
	for (FractureNetwork::ConstIterator fracture = fracture_network->begin(); fracture != fracture_network->end(); ++fracture, ++fid) {
		std::ostringstream file;
		file << "output/" << this->filename_base << "-frac" << fid << ".csv";
		std::ofstream ofs(file.str().c_str());
		ofs << "\"x\",\"y\",\"p\"" << std::endl;
		
		typename std::vector<Intersection<dim>>::const_iterator intersection = fracture->begin_intersection();
		for ( ; intersection != fracture->end_intersection(); ++intersection) {
			const Point<dim> x = fracture->vertex(intersection->index);
			ACI cell(this->triangulation, intersection->cells[0].first, intersection->cells[0].second, this->dof_handler);
			// If point is not exactly in this cell, then use the other cell
			if (intersection->n_cells > 1 && !(cell->point_inside(x)) ) {
				cell = ACI(this->triangulation, intersection->cells[1].first, intersection->cells[1].second, this->dof_handler);
				Assert(cell->point_inside(x), ExcMessage("Intersection point should be in either of the cells."));
			}
			Point<dim> x_loc = mapQ1.mapping.transform_real_to_unit_cell(cell, x);
			FEValues<dim> fe_values(*(this->fe), Quadrature<dim>(x_loc), update_values);
			fe_values.reinit(cell);
			fe_values.get_function_values(this->pressure_sol, pressure_val);
			ofs << x[0] << "," << x[1] << "," << pressure_val[0] << std::endl;
		}
		ofs.close();
	}
}


template <int dim>
void EllipticPressureSolver<dim>::output_system_matrix() const
{
	std::ofstream ofs("system_matrix.dat");
	this->system_matrix.print(ofs);
}


#endif // ELLIPTIC_PRESSURE_SOLVER_H