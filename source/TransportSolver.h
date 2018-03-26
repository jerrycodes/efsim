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

#ifndef TRANSPORT_SOLVER_H
#define TRANSPORT_SOLVER_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <iostream>
#include <cmath>

#include "HelpFunctions.h"
#include "ProblemFunctions.h"
#include "Velocity.h"

using namespace dealii;


/* Solve advection dominated transport problem with a DG(0) scheme.
 * Velocity can be taken from a function or calculated from a pressure solution.
 */
template<int dim>
class TransportSolver
{
public:
	TransportSolver(Triangulation<dim> &tria, const FE_DGQ<dim> &fe, DoFHandler<dim> &dh,
					const RockProperties<dim> &r, ProblemFunctionsTransport<dim> &f, ProblemType pt);

	void set_runtime_parameters(ParameterHandler& prm);
	void set_runtime_parameters(std::string filebame_base, unsigned int nq, double inflow_conc);
	
	void set_fractures(FractureNetwork& frac, double w);

	void run(double dt, double end_time,
			 VelocityData<dim>& velocity_in);

	void setup_system();

	void set_velocity(VelocityData<dim>& velocity_in);

	void solve_time_step(const double dt);

	double get_l2_error_norm() { return error_l2_norm;}
	double get_solution_min();
	double get_solution_max();
	double calculate_overshoot();
	void output_fracture_solution() const;
	
	double l2_error_reference_solution(std::string filename) const;

private:
	// Store the triangulation, finite element and dof handler as SmartPointers.
	// See Step 13 for further details.
	SmartPointer<Triangulation<dim>>  triangulation;
	SmartPointer<const FE_DGQ<dim>> 		fe_conc;
	SmartPointer<DoFHandler<dim>>     dh_conc;

	// Linear system variables
	SparsityPattern sparsity_pattern;
	SparseMatrix<double> mass_matrix;
	SparseMatrix<double> flux_matrix;
	SparseMatrix<double> system_matrix;
	Vector<double> system_rhs;
	Vector<double> flux_rhs;

	// Solution and old solution
	Vector<double> conc_sol;
	Vector<double> conc_sol_old;

	Vector<double> error;
	double error_l2_norm;
	double c_bound;
	double qoi_accumulated, qoi_accumulated_2;

	const RockProperties<dim>* rock;
	ProblemFunctionsTransport<dim>* funs;

	ProblemType problem_type;

	SmartPointer<VelocityData<dim>> velocity;

	SmartPointer<FractureNetwork> fracture_network;
	double fracture_width;
	
	double time;
	int time_step;

	// Input variables
	std::string filename_base;
	unsigned int n_qpoints;
	double inflow_concentration_value;
	unsigned int stride = 1;
	
	std::string qoi_file = "transport_qoi.dat";
	
	std::vector<std::pair<unsigned int, unsigned int>> qoi_cells;

	bool velocity_assembled = false;

	void assemble_time_dependent_system(const double dt);

	void assemble_velocity_dependent_system(VelocityData<dim>& velocity_in);

	void solve();
	void calculate_error();
	void calculate_qoi(const double dt);
	void init_qoi_file();

	void assemble_mass_matrix();
	void assemble_mass_matrix_fracture();

	void output_results() const;
	void output_solution_to_file() const;

	enum BoundaryIndicator {IN, OUT};
};


// Constructor takes triangulation, finite element, dof handler and problem type as input
template<int dim>
TransportSolver<dim>::TransportSolver(Triangulation<dim> &tria, const FE_DGQ<dim> &fe, DoFHandler<dim> &dh,
									  const RockProperties<dim> &r, ProblemFunctionsTransport<dim> &f, ProblemType pt)
	: triangulation(&tria),
	  fe_conc(&fe),
	  dh_conc(&dh),
	  rock(&r),
	  funs(&f),
	  problem_type(pt),
	  fracture_network(new FractureNetwork())
	  {}


// Set parameters from ParameterHandler
template <int dim>
void TransportSolver<dim>::set_runtime_parameters(ParameterHandler& prm)
{
	prm.enter_subsection("Transport solver");
	filename_base = prm.get("Output file base");
	inflow_concentration_value = prm.get_double("Inflow concentration");
	stride = prm.get_integer("Stride");
	prm.leave_subsection();
	prm.enter_subsection("Global");
	n_qpoints = prm.get_integer("No quadrature points");
	prm.leave_subsection();
}


// Set parameters
template <int dim>
void TransportSolver<dim>::set_runtime_parameters(std::string filename_base_in, unsigned int nq, double inflow_conc)
{
	filename_base = filename_base_in;
	n_qpoints = nq;
	inflow_concentration_value = inflow_conc;
}


template <int dim>
void TransportSolver<dim>::set_fractures(FractureNetwork& frac, double w)
{
	Assert(dim == 2 || frac.n_fractures() == 0, ExcNotImplemented());
	fracture_network = &frac;
	fracture_width = w;
}


/* Set up system:
 * - Initialize matrices, rhs and solutions vectors
 * - Assemble mass matrix (only dependent on grid)
 * - Set initial conditions
 * - Calculate initial error and output results
 */
template <int dim>
void TransportSolver<dim>::setup_system()
{
	dh_conc->distribute_dofs(*fe_conc);
	// The standard make_sparisty_pattern does not work in our case, as we use DG(0),
	// but still want face connections.

	DynamicSparsityPattern d_sparsity(dh_conc->n_dofs());
	GridTools::get_face_connectivity_of_cells<dim,dim>(*triangulation, d_sparsity);
	sparsity_pattern.copy_from(d_sparsity);

	funs->set_inflow_concentration(inflow_concentration_value);

	// Resize matrices
	mass_matrix.reinit(sparsity_pattern);
	system_matrix.reinit(sparsity_pattern);
	flux_matrix.reinit(sparsity_pattern);

	conc_sol.reinit(dh_conc->n_dofs());
	conc_sol_old.reinit(dh_conc->n_dofs());
	system_rhs.reinit(dh_conc->n_dofs());
	flux_rhs.reinit(dh_conc->n_dofs());

	// Assemble mass matrix (only dependent on grid)
	assemble_mass_matrix();

	// Set solution to initial values
	VectorTools::interpolate(*dh_conc, *(funs->initial_concentration), conc_sol_old);
	conc_sol = conc_sol_old;

	// Find upper bound on c due to maximum principle
	// TODO: Assumes well_conc <= inflow_concentration
	c_bound = *(std::max_element(conc_sol.begin(), conc_sol.end()));
	c_bound = std::max(c_bound, inflow_concentration_value);

	// Set t=0
	time = 0.0;
	time_step = 0;

	calculate_error();
	output_results();
	
	qoi_accumulated = 0.0;
	qoi_accumulated_2 = 0.0;
	init_qoi_file();
}


/* Set or update velocity field
 * Assembles part of system matrix that depends on velocity
 */
template <int dim>
void TransportSolver<dim>::set_velocity(VelocityData<dim>& velocity_in)
{
	velocity = &velocity_in;
	assemble_velocity_dependent_system(velocity_in);
}


/* Perform one time step of transport solver
 * Assumes that the velocity dependent matrix is already calculated
 */
template <int dim>
void TransportSolver<dim>::solve_time_step(const double dt)
{
	Assert(velocity_assembled, ExcMessage("You need to call set_pressure() before calling solve_time_step()"));

	time += dt;
	time_step++;

	std::cout << "Time step " << time_step << " (t = " << time << "s)" << std::endl;

	funs->set_time(time);

	assemble_time_dependent_system(dt);

	// Add mass matrix to system_matrix
	system_matrix.add(1.0, mass_matrix);
	system_matrix.add(dt, flux_matrix);
	system_rhs.add(dt, flux_rhs);

	// Add mass_matrix*old_solution to rhs
	mass_matrix.vmult_add(system_rhs, conc_sol_old);

	// Finally, solve linear system
	solve();
	calculate_error();
	calculate_qoi(dt);
	if ( (time_step % stride) == 0)
		output_results();

	std::cout << "  L2 error of concentration: " << get_l2_error_norm() << std::endl;
	std::cout << "  L2 norm of overshoot:      " << calculate_overshoot() << std::endl;
	std::cout << "  [min(c), max(c)]:          [" << get_solution_min() << ", " << get_solution_max() << "]" << std::endl;

	conc_sol_old = conc_sol;
}


/* Run transport until end_time with time step dt.
 * Assumes constant pressure and correction.
 */
template <int dim>
void TransportSolver<dim>::run(double dt, double end_time,
							   VelocityData<dim>& velocity_in)
{
	setup_system();
	set_velocity(velocity_in);
	// Time loop
	while (time < (end_time - dt/100)) {
		solve_time_step(dt);
	}
	output_solution_to_file();
	output_fracture_solution();
}


// Assemble mass matrix
template <int dim>
void TransportSolver<dim>::assemble_mass_matrix()
{
	const unsigned int dofs_per_cell = fe_conc->dofs_per_cell;
	std::vector<types::global_dof_index> cell_dofs(dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator
	cell = dh_conc->begin_active(),
	endc = dh_conc->end();
	for ( ; cell != endc; ++cell) {
		// Skip fracture cells
		if (cell->material_id() == 1)
			continue;
		cell->get_dof_indices(cell_dofs);
		const unsigned int cell_dof = cell_dofs[0];
		const double porosity_int = rock->get_poro(cell) * cell->measure();
		mass_matrix.add(cell_dof, cell_dof, porosity_int);
	}
	
	assemble_mass_matrix_fracture();
}


template <>
void TransportSolver<3>::assemble_mass_matrix_fracture()
{
	Assert(fracture_network->n_fractures() == 0, ExcNotImplemented());
}

template <>
void TransportSolver<2>::assemble_mass_matrix_fracture()
{
	// TODO: Use fracture porosity if in fracture
	typedef typename DoFHandler<2>::active_cell_iterator ACI;
	
	const unsigned int dofs_per_cell = fe_conc->dofs_per_cell;
	std::vector<types::global_dof_index> cell_dofs(dofs_per_cell);
	
	// Loop over fractures
	for (FractureNetwork::ConstIterator fracture = fracture_network->begin();
		 fracture != fracture_network->end(); ++fracture) {
		// Iterate through vertices defining fracture
		for (unsigned int vi=0; vi<fracture->n_vertices()-1; ++vi) {
			const Point<2> a = fracture->vertex(vi);
			const Point<2> b = fracture->vertex(vi+1);
			typename EmbeddedSurface<2>::IndexPair cell_info = fracture->get_segment_cell_info(vi);
			ACI cell = ACI(this->triangulation, cell_info.first, cell_info.second, dh_conc);
			cell->get_dof_indices(cell_dofs);
			const unsigned int cell_dof = cell_dofs[0];
			const double porosity_int = rock->get_poro(cell) * a.distance(b);
			mass_matrix.add(cell_dof, cell_dof, fracture_width * porosity_int);
		}
	}
}


// Assemble velocity dependent part of system matrix.
template <int dim>
void TransportSolver<dim>::assemble_velocity_dependent_system(VelocityData<dim>& velocity_in)
{
	// Set matrix and rhs to zero (keep sparsity pattern)
	flux_matrix = 0;
	flux_rhs = 0;

	typename VelocityData<dim>::FE_Pointer fe_velocity = velocity_in.get_fe();
	typename VelocityData<dim>::DH_Pointer dh_velocity = velocity_in.get_dh();

	FEFaceValues<dim> fe_face_values_velocity(*fe_velocity, QGauss<dim-1>(n_qpoints),
											  update_values | update_quadrature_points |
											  update_normal_vectors | update_JxW_values);

	const unsigned int dofs_per_cell  = fe_conc->dofs_per_cell;
	const unsigned int n_qpoints_face = fe_face_values_velocity.n_quadrature_points;

	std::vector<types::global_dof_index> cell_dofs(dofs_per_cell);
	std::vector<types::global_dof_index> neighbor_dofs(dofs_per_cell);

	std::vector<Vector<double>> velocity_values(n_qpoints_face, Vector<double>(dim));

	typename DoFHandler<dim>::active_cell_iterator
	cellc = dh_conc->begin_active(),
	cellv = dh_velocity->begin_active(),
	endc  = dh_conc->end();

	for ( ; cellc != endc; ++cellc, ++cellv) {
		cellc->get_dof_indices(cell_dofs);
		const unsigned int cell_dof = cell_dofs[0];
		// Loop through faces
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			fe_face_values_velocity.reinit(cellv, face);
			velocity_in.get_values(fe_face_values_velocity, velocity_values);
			// Boundary faces
			if (cellc->at_boundary(face)) {
				for (unsigned int q=0; q<n_qpoints_face; ++q) {
					const double face_flux = velocity_in.scalar_product(velocity_values[q], fe_face_values_velocity.normal_vector(q));
					if ( face_flux < 0 ) {
						// Inflow
						Point<dim> qpoint = fe_face_values_velocity.quadrature_point(q);
						flux_rhs(cell_dof) -= funs->boundary_concentration->value(qpoint, 0) * face_flux * fe_face_values_velocity.JxW(q);
					}
					else {
						// Outflow
						flux_matrix.add(cell_dof, cell_dof, face_flux * fe_face_values_velocity.JxW(q));
					}
				}
			}
			// Internal faces
			else if ( cellc->face(face)->has_children() ) {
				// visit this from the other side
				continue;
			}
			else {
				cellc->neighbor(face)->get_dof_indices(neighbor_dofs);
				const unsigned int neighbor_dof = neighbor_dofs[0];
				for (unsigned int q=0; q<n_qpoints_face; ++q) {
					double face_flux = velocity_in.scalar_product(velocity_values[q], fe_face_values_velocity.normal_vector(q));
					if ( face_flux > 0) {
						// Upwind
						flux_matrix.add(cell_dof, cell_dof, face_flux * fe_face_values_velocity.JxW(q));
						flux_matrix.add(neighbor_dof, cell_dof, -face_flux * fe_face_values_velocity.JxW(q));
					}
					else if ( cellc->neighbor_is_coarser(face) ) {
						// If neighbor is coarser, visit it now (only if current cell is not upwind)
						face_flux = -face_flux;
						flux_matrix.add(neighbor_dof, neighbor_dof, face_flux * fe_face_values_velocity.JxW(q));
						flux_matrix.add(cell_dof, neighbor_dof, -face_flux * fe_face_values_velocity.JxW(q));
					}
				}
			}
		}
	}
	velocity_assembled = true;
}


// Asemble time dependent part of linear system.
template <int dim>
void TransportSolver<dim>::assemble_time_dependent_system(const double dt)
{
	system_matrix = 0;
	system_rhs = 0;

	FEValues<dim> fe_values_concentration(*fe_conc, QGauss<dim>(n_qpoints),
										  update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell  = fe_conc->dofs_per_cell;
	const unsigned int nq      = fe_values_concentration.n_quadrature_points;

	std::vector<types::global_dof_index> cell_dofs(dofs_per_cell);
	std::vector<types::global_dof_index> neighbor_dofs(dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator
	cell = dh_conc->begin_active(),
	endc = dh_conc->end();

	for ( ; cell != endc; ++cell) {
		fe_values_concentration.reinit(cell);
		cell->get_dof_indices(cell_dofs);
		const unsigned int cell_dof = cell_dofs[0];

		// Integrate source
		double rhs_minus = 0.0;
		double rhs_plus  = 0.0;
		for (unsigned int q=0; q<nq; ++q) {
			const Point<dim> q_point = fe_values_concentration.quadrature_point(q);
			rhs_minus    += std::min(funs->right_hand_side->value(q_point, 0), 0.0) * fe_values_concentration.JxW(q);
			rhs_plus     += funs->well_concentration->value(q_point) * std::max(funs->right_hand_side->value(q_point, 0), 0.0)
							* fe_values_concentration.JxW(q);
		}

		// TODO: Fix this so that only RightHandSideTransport is needed (include WellConcentration insise value())
		// If we have a source term (RightHandSideTransport) that is constructed from an analytic
		// solution, set rhs = 0 here.
		if (problem_type == ProblemType::SIMPLE_ANALYTIC || problem_type == ProblemType::ANALYTIC) {
			rhs_minus = 0.0;
			rhs_plus = 0.0;
		}

		system_matrix.add(cell_dof, cell_dof, dt*rhs_minus);

		system_rhs(cell_dof) += dt*rhs_plus;

		if (problem_type == ProblemType::SIMPLE_ANALYTIC || problem_type == ProblemType::ANALYTIC) {
			// Add time dependent source
			double rhs_int = 0.0;
			for (unsigned int q=0; q<nq; ++q) {
				const Point<dim> q_point = fe_values_concentration.quadrature_point(q);
				rhs_int += funs->right_hand_side_transport->value(q_point) * fe_values_concentration.JxW(q);
			}
			system_rhs(cell_dof) += dt*rhs_int;
		}
	}
}


// Set up a CG solver and solve linear system
template <int dim>
void TransportSolver<dim>::solve()
{
	// TODO: Initialize solver only once? (memeber function)
	SolverControl solver_control (dh_conc->n_dofs(), 1e-10, false, false);
	SolverGMRES<> solver(solver_control);
	PreconditionSSOR<SparseMatrix<double>> precondition;
	precondition.initialize(system_matrix, 1.0);
	solver.solve(system_matrix, conc_sol, system_rhs,
				 precondition);
}


// Calculate error in concentration and its L2 norm
template <int dim>
void TransportSolver<dim>::calculate_error()
{
	funs->exact_concentration->set_time(time);
	Vector<double> exact_sol_vec(triangulation->n_active_cells());
	VectorTools::interpolate(*dh_conc, *(funs->exact_concentration), exact_sol_vec);
	error = conc_sol;
	error -= exact_sol_vec;
	for (unsigned int i=0; i<error.size(); ++i)
		error(i) = abs(error(i));
	Vector<double> error_norm_per_cell(triangulation->n_active_cells());
	VectorTools::integrate_difference(*dh_conc, conc_sol, *(funs->exact_concentration),
									  error_norm_per_cell, QGauss<dim>(n_qpoints +1),
									  VectorTools::L2_norm);
	error_l2_norm = error_norm_per_cell.l2_norm();
}


template <int dim>
void TransportSolver<dim>::init_qoi_file()
{
	std::string header;
	
	switch (problem_type)
	{
		case SIMPLE_FRAC_RESOLVED:
			header = "Time\tFlux Out\tAccumulated";
			break;
		case REGULAR_NETWORK:
		case REGULAR_NETWORK_RESOLVED:
			header = "Time\tFlux out (y=0.5)\tFlux out (y=0.75)";
			break;
		default:
			header = "(No QOI data specified in TransportSolver)";
	}
	
	std::ofstream ofs(qoi_file, std::ofstream::trunc);
	ofs << header << std::endl;
	ofs.close();
	
	// Find cells on fracture outflow for resolved Flemisch case
	if (problem_type == REGULAR_NETWORK_RESOLVED) {
		const unsigned int face = 1;
		const double half_width = 1e-4/2.0;
		typename DoFHandler<dim>::active_cell_iterator
		cell = dh_conc->begin_active(),
		endc = dh_conc->end();
		for ( ; cell != endc; ++cell) {
			if (cell->at_boundary(face)) {
				Point<dim> face_center = cell->face(face)->center();
				if ( (face_center(1) > 0.5 - half_width) && (face_center(1) < 0.5 + half_width) )
					qoi_cells.push_back(std::pair<unsigned int,unsigned int >(cell->level(), cell->index()));
				else if ( (face_center(1) > 0.75 - half_width) && (face_center(1) < 0.75 + half_width) )
					qoi_cells.push_back(std::pair<unsigned int, unsigned int>(cell->level(), cell->index()));
			}
		}
	}
}


template <int dim>
void TransportSolver<dim>::calculate_qoi(const double dt)
{
	std::ofstream ofs(qoi_file, std::ofstream::app);
	
	// Choose QOI based on problem type
	switch (problem_type)
	{
		case SIMPLE_FRAC_RESOLVED:
		{
			Assert(dim == 2, ExcNotImplemented());
			
			// Integrate flux out near fracture
			double fracture_flux = 0.0;
			typename DoFHandler<dim>::active_cell_iterator
			cell = dh_conc->begin_active(),
			endc = dh_conc->end();
			for ( ; cell != endc; ++cell) {
				if (cell->at_boundary(1)) {
					Point<dim> face_center = cell->face(1)->center();
					if (face_center(1) > 9.0/16.0 && face_center(1) < 10.0/16.0) {
						const unsigned int dof = cell->dof_index(0);
						fracture_flux += cell->face(1)->measure() * 0.5 * (conc_sol(dof) + conc_sol_old(dof)) * dt;
					}
				}
			}
			
			qoi_accumulated += fracture_flux;
			
			ofs << time << "\t" << fracture_flux << "\t" << qoi_accumulated << std::endl;
			
			break;
		}
		
		case REGULAR_NETWORK:
		{
			Assert(dim == 2, ExcNotImplemented());
			
			typedef typename DoFHandler<dim>::active_cell_iterator ACI;
			
			const unsigned int face_right = 1;
			std::vector<ACI> cells;
			std::vector<double> conc_flux_out(2);
			Assert(fracture_network->n_fractures() > 0, ExcInternalError());
			FractureNetwork::ConstIterator fracture = fracture_network->begin();
			EmbeddedSurface<2>::IndexPair cell_info = fracture->get_segment_cell_info(fracture->n_vertices()-2);
			cells.push_back(ACI(triangulation, cell_info.first, cell_info.second, dh_conc));
			fracture += 2;
			cell_info = fracture->get_segment_cell_info(fracture->n_vertices()-2);
			cells.push_back(ACI(triangulation, cell_info.first, cell_info.second, dh_conc));
			
			for (unsigned int i=0; i<2; ++i) {
				Assert(cells[i]->at_boundary(face_right), ExcMessage("Should be at boundary"));
				ACI cell_velocity(triangulation, cells[i]->level(), cells[i]->index(), velocity->get_dh());
				std::vector<unsigned int> velocity_dofs(4);
				cell_velocity->get_dof_indices(velocity_dofs);
				const double flux_out = velocity->get_dof_value(velocity_dofs[face_right]);
				const unsigned int dof = cells[i]->dof_index(0);
				conc_flux_out[i] = flux_out * 0.5 * (conc_sol(dof) + conc_sol_old(dof)) * dt;
			}
			
			qoi_accumulated += conc_flux_out[0];
			qoi_accumulated_2 += conc_flux_out[1];
			
			ofs << time << "\t" << conc_flux_out[0] << "\t" << qoi_accumulated << "\t" << conc_flux_out[1] << "\t" << qoi_accumulated_2 << std::endl;
			
			break;
		}
		
		case REGULAR_NETWORK_RESOLVED:
		{
			Assert(dim == 2, ExcNotImplemented());
			typedef typename DoFHandler<dim>::active_cell_iterator ACI;
			const unsigned int face_right = 1;
			
			std::vector<double> conc_flux_out(2);
			
			for (unsigned int i=0; i<qoi_cells.size(); ++i) {
				ACI cell(triangulation, qoi_cells[i].first, qoi_cells[i].second, dh_conc);
				const unsigned int dof = cell->dof_index(0);
				ACI cell_velocity(triangulation, qoi_cells[i].first, qoi_cells[i].second, velocity->get_dh());
				std::vector<unsigned int> velocity_dofs(4);
				cell_velocity->get_dof_indices(velocity_dofs);
				const double flux_out_integrated = velocity->get_dof_value(velocity_dofs[face_right]);
				if (cell->center()[1] < 0.6)
					conc_flux_out[0] += flux_out_integrated * 0.5 * (conc_sol(dof) + conc_sol_old(dof)) * dt;
				else
					conc_flux_out[1] += flux_out_integrated * 0.5 * (conc_sol(dof) + conc_sol_old(dof)) * dt;
			}
			
			qoi_accumulated += conc_flux_out[0];
			qoi_accumulated_2 += conc_flux_out[1];
			
			ofs << time << "\t" << conc_flux_out[0] << "\t" << qoi_accumulated << "\t" << conc_flux_out[1] << "\t" << qoi_accumulated_2 << std::endl;
			
			break;
		}
		
		default:
		{
			// Do nothing
		}
	}
	
	ofs.close();
}


// Output concentration and error to vtk file
template <int dim>
void TransportSolver<dim>::output_results() const
{
	DataOut<dim> data_out;

	data_out.attach_dof_handler(*dh_conc);
	data_out.add_data_vector(conc_sol, "concentration", DataOut<dim>::DataVectorType::type_dof_data);
	data_out.add_data_vector(error, "error", DataOut<dim>::DataVectorType::type_dof_data);

	data_out.build_patches();

	std::ostringstream filename;
    filename << "output/" << filename_base << "-" << time_step << ".vtk";
    std::ofstream output(filename.str().c_str());
	data_out.write_vtk(output);
}


template <int dim>
void TransportSolver<dim>::output_solution_to_file() const
{
    std::ofstream output("output/conc_sol.vec");
	conc_sol.block_write(output);
}


template <>
void TransportSolver<3>::output_fracture_solution() const
{
	Assert(false, ExcNotImplemented());
}


template <>
void TransportSolver<2>::output_fracture_solution() const
{
	const unsigned int dim = 2;
	typedef typename DoFHandler<dim>::active_cell_iterator ACI;
	typedef typename EmbeddedSurface<dim>::IndexPair IP;
	
	Point<dim> x0;
	Point<dim> x1;
	
	unsigned int fid = 1;
	for (FractureNetwork::ConstIterator fracture = fracture_network->begin(); fracture != fracture_network->end(); ++fracture, ++fid) {
		std::ostringstream file;
		file << "output/" << filename_base << "-frac" << fid << ".csv";
		std::ofstream ofs(file.str().c_str());
		ofs << "\"x0\",\"y0\",\"x1\",\"y1\",\"c\"" << std::endl;
		
		x0 = fracture->vertex(0);
		
		typename std::vector<Intersection<dim>>::const_iterator intersection = fracture->begin_intersection();
		for ( ; intersection != fracture->end_intersection(); ++intersection) {
			x1 = fracture->vertex(intersection->index);
			const IP cell_info = intersection->cells[0];
			const ACI cell(triangulation, cell_info.first, cell_info.second, dh_conc);
			const double conc = conc_sol[cell->dof_index(0)];
			
			ofs << x0[0] << "," << x0[1] << "," << x1[0] << "," << x1[1] << "," << conc << std::endl;
			
			x0 = x1;
		}
		
		// Handle last segment
		x1 = fracture->vertex(fracture->n_vertices()-1);
		const IP cell_info = fracture->get_segment_cell_info(fracture->n_vertices()-2);
		const ACI cell(triangulation, cell_info.first, cell_info.second, dh_conc);
		const double conc = conc_sol[cell->dof_index(0)];
		ofs << x0[0] << "," << x0[1] << "," << x1[0] << "," << x1[1] << "," << conc << std::endl;
		
		ofs.close();
	}
}


template <int dim>
double TransportSolver<dim>::l2_error_reference_solution(string filename) const
{
	std::ifstream ref_sol_file(filename);
	if (ref_sol_file.is_open()) {
		Vector<double> reference_solution;
		reference_solution.block_read(ref_sol_file);
		if (reference_solution.size() != dh_conc->n_dofs()) {
			std::cout << "Warning. Reference solution file <" << filename << "> contains wrong number of elements." << std::endl;
			return NaN;
		}
		double l2_error_squared = 0.0;
		typename DoFHandler<dim>::active_cell_iterator
		cell = dh_conc->begin_active(),
		endc = dh_conc->end();
		for ( ; cell != endc; ++cell) {
			const unsigned int dof = cell->dof_index(0);
			l2_error_squared += pow(reference_solution[dof] - conc_sol[dof], 2) * cell->measure();
		}
		return sqrt(l2_error_squared);
	}
	else {
		std::cout << "Warning. Unable to open reference solution file <" << filename << ">." << std::endl;
		return NaN;
	}
}



// Get min of concentration
template <int dim>
double TransportSolver<dim>::get_solution_min()
{
	Vector<double>::const_iterator min = std::min_element(conc_sol.begin(), conc_sol.end());
	return *min;
}


// Get min of concentration
template <int dim>
double TransportSolver<dim>::get_solution_max()
{
	Vector<double>::const_iterator max = std::max_element(conc_sol.begin(), conc_sol.end());
	return *max;
}


// ||max(c-c^, 0) + max(-c, 0)||
template <int dim>
double TransportSolver<dim>::calculate_overshoot()
{
	FEValues<dim> fe_values(*fe_conc, QGauss<dim>(n_qpoints), update_values |
							update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell  = fe_conc->dofs_per_cell;

	std::vector<types::global_dof_index> cell_dofs(dofs_per_cell);
	
	typename DoFHandler<dim>::active_cell_iterator
	cell = dh_conc->begin_active(),
	endc = dh_conc->end();

	double overshoot = 0.0;
	for ( ; cell != endc; ++cell) {
		fe_values.reinit(cell);
		cell->get_dof_indices(cell_dofs);
		const unsigned int cell_dof = cell_dofs[0];
		const double conc_cell = conc_sol(cell_dof);
		const double overshoot_cell = std::max(conc_cell-c_bound, 0.0) + std::max(-conc_cell, 0.0);
		overshoot += cell->measure() * overshoot_cell * overshoot_cell;
	}
	return std::sqrt(overshoot);
}


#endif // TRANSSPORT_SOLVER_H
