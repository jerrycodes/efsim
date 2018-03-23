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

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/numerics/data_out.h>

#include "ProblemFunctions.h"
#include "RockProperties.h"
#include "HelpFunctions.h"
#include "EmbeddedSurface.h"


using namespace dealii;

#ifndef VELOCITY_H
#define VELOCITY_H


// Class to store velocity data


template <int dim>
class VelocityData : public Subscriptor
{
public:
	VelocityData(const Triangulation<dim> &tria)
	: Subscriptor(), degree(0), fe(degree), dh(tria) {}
	
	void setup_system()
	{
		dh.distribute_dofs(fe);
		dof_values.reinit(dh.n_dofs());
		fracture_intersection_dof = std::vector<bool>(dh.n_dofs(),false);
	}
	
	void init_exact(const ProblemFunctionsFlow<dim> flow_fun);
	
	double calculate_residuals(Vector<double> &residuals,
							   const Function<dim>* source,
							   const FiniteElement<dim>* fe_pressure = new FE_Nothing<dim>(),
							   const DoFHandler<dim>* dh_pressure = NULL,
							   const Vector<double> pressure_diff_scaled = Vector<double>());
	
	double calculate_residuals(Vector<double> &residuals,
							   const ProblemFunctionsFlow<dim> flow_fun,
							   const FractureNetwork fractures);
	
	std::pair<double,double> calculate_flux_error(const RockProperties<dim> rock, TensorFunction<1,dim>* exact_gradient) const;
	
	void add_correction(const Vector<double> correction);
	
	void get_values(const FEValuesBase<dim>& fe_values, std::vector<Vector<double>>& result) const;
	void get_divergences(const FEValuesBase<dim>& fe_values, std::vector<double>& result) const;
	
	double get_dof_value(unsigned int i, bool zero_on_fracture = false) const 
	{return (zero_on_fracture && fracture_intersection_dof[i]) ? 0.0 : dof_values[i]; } 
	
	Tensor<1,dim+1> get_value(const typename DoFHandler<dim>::active_cell_iterator cell, const Point<dim> p_loc);
	double get_divergence(const typename DoFHandler<dim>::active_cell_iterator cell, const Point<dim> p_loc);
	
	void set_constant_fracture_velocity(const FractureNetwork fractures, double velocity);
	
	bool is_locally_conservative() { return locally_conservative; }
	
	void set_dof_value(unsigned int dof, const double value)
	{
		AssertIndexRange(dof, dh.n_dofs());
		dof_values(dof) = value;
	}
	
	void add_to_dof_value(unsigned int dof, const double value)
	{
		AssertIndexRange(dof, dh.n_dofs());
		dof_values(dof) += value;
	}
	
	void add_fracture_intersection_dof(unsigned int dof)
	{ fracture_intersection_dof[dof] = true; }
	
	void apply_constraints();
	
	typedef const SmartPointer<FE_RaviartThomas<dim>> FE_Pointer;
	typedef const SmartPointer<DoFHandler<dim>>       DH_Pointer;
	DH_Pointer get_dh() { return DH_Pointer(&dh); }
	FE_Pointer get_fe() { return FE_Pointer(&fe); }
	
	void write_to_vtk(std::string file_base);
	void output_fracture_velocity(const FractureNetwork* fractures, double width) const;
	
	double scalar_product(Vector<double> v, Tensor<1,dim,double> p);
	
private:
	const unsigned int degree;
	FE_RaviartThomas<dim> fe;
	DoFHandler<dim>       dh;
	Vector<double>        dof_values;
	
	ConstraintMatrix constraints;
	
	bool locally_conservative;
	bool globally_conservative;
	
	unsigned int n_qpoints = 2;
	double tol = 1e-10;
	
	std::vector<bool> fracture_intersection_dof;
	
	Point<dim> vector_to_point(Vector<double> vec) const;
};


template <int dim>
void VelocityData<dim>::init_exact(const ProblemFunctionsFlow<dim> flow_fun)
{
	// TODO: This function assumes perm = 1
	FEFaceValues<dim> fe_face_values(fe, QGauss<dim-1>(2),
									 update_quadrature_points | update_normal_vectors | update_JxW_values);
	std::vector<unsigned int> face_dofs(fe.n_dofs_per_cell());
	const unsigned int n_qpoints_face = fe_face_values.n_quadrature_points;
	
	typename DoFHandler<dim>::active_cell_iterator
	cell = dh.begin_active(),
	endc = dh.end();
	for ( ; cell!=endc; ++cell) {
		cell->get_dof_indices(face_dofs);
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			fe_face_values.reinit(cell, face);
			double face_flux = 0.0;
			for (unsigned int q=0; q<n_qpoints_face; ++q)
				face_flux += -flow_fun.exact_gradient->value(fe_face_values.quadrature_point(q)) * fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
			dof_values(face_dofs[face]) = GeometryInfo<dim>::unit_normal_orientation[face] * face_flux;
		}
	}
}


template <int dim>
void VelocityData<dim>::apply_constraints()
{
	constraints.clear();
	setup_flux_constraints_subfaces(dh, constraints);
	constraints.close();
	constraints.distribute(dof_values);
	constraints.distribute(fracture_intersection_dof);
}


DEAL_II_NAMESPACE_OPEN

template <>
void ConstraintMatrix::distribute(std::vector<bool> &vec) const
{
	for (size_type i=0; i!=lines.size(); ++i) {
		for (size_type j=0; j<lines[i].entries.size(); ++j) {
			if (! vec[lines[i].line])
				vec[lines[i].line] = vec[lines[i].entries[j].first];
		}
	}
}

DEAL_II_NAMESPACE_CLOSE


template <int dim>
void VelocityData<dim>::set_constant_fracture_velocity(const FractureNetwork fractures, double velocity)
{
	typedef typename DoFHandler<dim>::active_cell_iterator ACI;
	typedef typename Intersection<dim>::IndexPair IP;
	
	std::vector<unsigned int> face_dofs(fe.n_dofs_per_cell());
	
	for (FractureNetwork::ConstIterator fracture = fractures.begin(); fracture != fractures.end(); ++fracture) {
		typename std::vector<Intersection<dim>>::const_iterator intersection = fracture->begin_intersection();
		for ( ; intersection != fracture->end_intersection(); ++intersection) {
			
			unsigned int loc_cell;
			int direction;
			if (intersection->n_cells == 1) {
				loc_cell = 0;
				direction = (intersection->index == 0) ? -1 : 1;
			}
			else {
				Assert(intersection->n_cells == 2, ExcInternalError());
				if (intersection->cells[0].first > intersection->cells[1].first) {
					loc_cell = 0;
					direction = 1;
				}
				else {
					loc_cell = 1;
					direction = -1;
				}
			}
			
			ACI cell(&(dh.get_triangulation()), intersection->cells[loc_cell].first, intersection->cells[loc_cell].second, &dh);
			unsigned int face = intersection->faces[loc_cell];
			
			cell->get_dof_indices(face_dofs);
			const int orientation = direction * GeometryInfo<dim>::unit_normal_orientation[face];
			set_dof_value(face_dofs[face], orientation * velocity);
			add_fracture_intersection_dof(face_dofs[face]);
		}
	}
	apply_constraints();
}


/*
 * Calculate residuals
 * Return L2 norm of the residual.
 */
template <int dim>
double VelocityData<dim>::calculate_residuals(Vector<double> &residuals,
											  const Function<dim>* source,
											  const FiniteElement<dim>* fe_pressure,
											  const DoFHandler<dim>* dh_pressure,
											  const Vector<double> pressure_diff_scaled)
{
	residuals = 0;
	
	bool time_dependent = true;
	if (dh_pressure == NULL) {
		Assert(pressure_diff_scaled.size() == 0, ExcMessage("This vector should be empty when dh_correction points to NULL"));
		time_dependent = false;
	}
	
	FEValues<dim>     fe_values(fe, QGauss<dim>(n_qpoints), update_values | update_quadrature_points | update_JxW_values);
	FEValues<dim>     fe_values_pressure(*fe_pressure, QGauss<dim>(n_qpoints), update_values | update_JxW_values);
	
	const unsigned int n_qpoints_cell = fe_values.n_quadrature_points;
	
	double residual_sum = 0.0;
	double residual_l2norm = 0.0;
	
	Assert (fe.n_dofs_per_cell() == GeometryInfo<dim>::faces_per_cell, ExcInternalError());
	std::vector<unsigned int> face_dofs(fe.n_dofs_per_cell());
	
	locally_conservative = true;
	globally_conservative = false;
	
	typename DoFHandler<dim>::active_cell_iterator
	cell = dh.begin_active(),
	endc = dh.end(),
	cellp;
	if (time_dependent) cellp = dh_pressure->begin_active();
	unsigned int ci = 0;
	for ( ; cell!=endc; ++cell, ++ci) {
		fe_values.reinit(cell);
		
		// Integrate source over cell
		double source_integrated = 0.0;
		for (unsigned int q=0; q<n_qpoints_cell; ++q) {
			source_integrated += source->value(fe_values.quadrature_point(q)) * fe_values.JxW(q);
		}
		
		if (time_dependent) {
			std::vector<double> pressure_diff_values(n_qpoints_cell);
			fe_values_pressure.reinit(cellp);
			fe_values_pressure.get_function_values(pressure_diff_scaled, pressure_diff_values);
			double pressure_update_integrated = 0.0;
			for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q) {
				pressure_update_integrated += pressure_diff_values[q] * fe_values.JxW(q);
			}
			source_integrated -= pressure_update_integrated;
		}
		
		// TODO: Use sum_dofs here when code is well tested
		cell->get_dof_indices(face_dofs);
		double flux_out = 0.0;
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			flux_out += GeometryInfo<dim>::unit_normal_orientation[face] * dof_values[face_dofs[face]];
		}
		
		// Check if flux is conservative over cell
		if (abs(source_integrated - flux_out) > tol )
			locally_conservative = false;
		residuals(ci) += (source_integrated - flux_out) / cell->measure();
		residual_l2norm += pow(residuals(ci), 2.0) * cell->measure();
		residual_sum += source_integrated - flux_out;
		
		if (time_dependent) ++cellp;
	}
	
	residual_l2norm = sqrt(residual_l2norm);
	
	// TODO: This tolerance seems to be to fine when we have hanging nodes
	if (abs(residual_sum) < tol)
		globally_conservative = true;

	if (globally_conservative)
		std::cout << "Velocity is globally conservative!" << std::endl;
	else
		std::cout << "Velocity is NOT globally conservative! (Sum residuals = " << residual_sum << ")" << std::endl;

	if (locally_conservative)
		std::cout << "Velocity is locally conservative!" << std::endl;
	else
		std::cout << "Velocity is NOT locally conservative! (L2-norm residuals = " << residual_l2norm << ")" << std::endl;

	return residual_l2norm;
}


template <>
double VelocityData<3>::calculate_residuals(Vector<double> &,
											const ProblemFunctionsFlow<3>,
											const FractureNetwork)
{
	Assert(false, ExcNotImplemented());
	return NaN;
}

template <>
double VelocityData<2>::calculate_residuals(Vector<double> &residuals,
											const ProblemFunctionsFlow<2> flow_fun,
											const FractureNetwork fractures)
{
	// TODO: Does not work for non-zero rhs
	
	const unsigned int dim = 2;
	
	double residual_l2norm = calculate_residuals(residuals, flow_fun.right_hand_side);
	if (fractures.n_fractures() == 0)
		return residual_l2norm;
	
	typedef typename Triangulation<dim>::active_cell_iterator ACI;
	
	// Loop over fractures
	for (FractureNetwork::ConstIterator fracture = fractures.begin(); fracture != fractures.end(); ++fracture) {
		
		// Iterate through vertices defining fracture
		for (unsigned int vi=0; vi<fracture->n_vertices()-1; ++vi) {
			const Point<dim> a = fracture->vertex(vi);
			const Point<dim> b = fracture->vertex(vi+1);
			typename EmbeddedSurface<dim>::IndexPair cell_info = fracture->get_segment_cell_info(vi);
			ACI cell = ACI(&(dh.get_triangulation()), cell_info.first, cell_info.second);
			unsigned int ci = cell->active_cell_index();
			const Point<dim> midpoint = (a+b)/2.0;
			const double rhs_value = flow_fun.right_hand_side_fracture->value(midpoint,0);
			residuals(ci) += (rhs_value * a.distance(b)) / cell->measure();
		}
	}
	
	residual_l2norm = 0.0;
	typename DoFHandler<dim>::active_cell_iterator
	cell = dh.begin_active(),
	endc = dh.end();
	unsigned int ci = 0;
	for ( ; cell!=endc; ++cell, ++ci) {
		residual_l2norm += pow(residuals(ci), 2.0) * cell->measure();
	}
	
	return sqrt(residual_l2norm);
}


template <int dim>
std::pair<double,double> VelocityData<dim>::calculate_flux_error(const RockProperties<dim> rock, TensorFunction<1,dim>* exact_gradient) const
{
	FEFaceValues<dim> fe_face_values(fe, QGauss<dim-1>(n_qpoints), 
									 update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);
	
	const unsigned int n_qpoints_face = fe_face_values.n_quadrature_points;

	double flux_error_l2norm   = 0.0;
	double flux_error_edgenorm = 0.0;
	
	Assert (fe.n_dofs_per_cell() == GeometryInfo<dim>::faces_per_cell, ExcInternalError());
	
	std::vector<Vector<double>> velocity_values(n_qpoints_face, Vector<double>(dim));
	
	typename DoFHandler<dim>::active_cell_iterator
	cell = dh.begin_active(),
	endc = dh.end();
	unsigned int ci = 0;
	for ( ; cell!=endc; ++cell, ++ci) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			// If neighbors are finer: skip and visit from other side
			if ( cell->face(face)->has_children() )
				continue;
			
			fe_face_values.reinit(cell, face);
			const double face_meas = face_measure<dim>(cell, face);
			fe_face_values.get_function_values(dof_values, velocity_values);
			
			const Tensor<2,dim> perm = rock.get_perm(cell);
			
			for (unsigned int q=0; q<n_qpoints_face; ++q) {
				const Tensor<1,dim,double> velocity_exact = -perm*exact_gradient->value(fe_face_values.quadrature_point(q));
				const double flux_diff = (vector_to_point(velocity_values[q]) - velocity_exact) * fe_face_values.normal_vector(q);
				double integrand = pow(flux_diff, 2.0) * fe_face_values.JxW(q);
				// If neighbor is equally coarse (remember, we already checked if neighbor is finer longer up)
				// multiply by 1/2 since we visit this face from both sides.
				if ( !(cell->at_boundary(face)) && !(cell->neighbor_is_coarser(face)) )
					integrand *= 0.5;
				flux_error_l2norm += integrand;
				flux_error_edgenorm += face_meas * integrand;
			}
		}
	}
	flux_error_l2norm = sqrt(flux_error_l2norm);
	flux_error_edgenorm = sqrt(flux_error_edgenorm);
	
	return std::pair<double,double>(flux_error_l2norm, flux_error_edgenorm);
}


template <int dim>
void VelocityData<dim>::add_correction(const Vector<double> correction)
{
	// TODO: If correction is multiplied by face measure in postprocessor, this routine would be simple, just add correction to dof_values.
	AssertDimension(correction.size(), dh.n_dofs());
	
	std::vector<unsigned int> face_dofs(fe.n_dofs_per_cell());
		
	typename DoFHandler<dim>::active_cell_iterator
	cell = dh.begin_active(),
	endc = dh.end();
	for ( ; cell!=endc; ++cell) {
		cell->get_dof_indices(face_dofs);
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			// If neighbor is equally coarse and alreadu visited, jump to next 
			if ( !(cell->at_boundary(face)) ) {
				if ( (cell->level() == cell->neighbor_level(face)) && cell->neighbor(face)->user_flag_set() ) {
					continue;
				}
			}
			const double face_meas = face_measure<dim>(cell, face);
			const unsigned int dof = face_dofs[face];
			dof_values[dof] -= face_meas * correction[dof];
		}
		cell->set_user_flag();
	}
}


template <int dim>
void VelocityData<dim>::get_values(const FEValuesBase<dim>& fe_values, std::vector<Vector<double>>& result) const
{
	unsigned int n_qpoints = fe_values.n_quadrature_points;
	
	// Check if we have the same FE. OBS! Costly operation.
	Assert(fe_values.get_fe() == fe, ExcMessage("Finite elements should be equal!"));
	
	// Check if dimension of result vector is equal to number of quadrature points and each component has dim entries
	AssertVectorVectorDimension(result, n_qpoints, dim);
	
	fe_values.get_function_values(dof_values, result);
}


template <int dim>
void VelocityData<dim>::get_divergences(const FEValuesBase<dim>& fe_values, std::vector<double>& result) const
{
	unsigned int n_qpoints = fe_values.n_quadrature_points;
	
	// Check if we have the same FE. OBS! Costly operation.
	Assert(fe_values.get_fe() == fe, ExcMessage("Finite elements should be equal!"));
	
	// Check if dimension of result vector is equal to number of quadrature points
	AssertDimension(result.size(), n_qpoints);
	
	FEValuesViews::Vector<dim> fe_values_view(fe_values, 0);
	fe_values_view.get_function_divergences(dof_values, result);
}


template <int dim>
Tensor<1,dim+1> VelocityData<dim>::get_value(const typename DoFHandler<dim>::active_cell_iterator cell, const Point<dim> p_loc)
{
	Assert(degree == 0, ExcMessage("Function assumes RT(0) space"));
	Tensor<1,dim+1> result;
	Vector<double> cell_dof_values(fe.dofs_per_cell);
	cell->get_dof_values(dof_values, cell_dof_values);
	for (unsigned int i=0; i<fe.dofs_per_cell; ++i) {
		// Need to divide cell_dof_values by corresponding face measure
		cell_dof_values[i] /= cell->face(i)->measure();
		for (int d=0; d<dim; ++d)
			result[d] += cell_dof_values[i] * fe.shape_value_component(i, p_loc, d);
	}
	result[dim] = 1.0;
	return result;
}


template <int dim>
void VelocityData<dim>::write_to_vtk(std::string file_base)
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dh);
	
	std::vector<DataComponentInterpretation::DataComponentInterpretation> 
		component_type(dim, DataComponentInterpretation::component_is_part_of_vector);
	data_out.add_data_vector(dh, dof_values, "velocity", component_type);
	
	std::ostringstream file_name;
	file_name << "output/" << file_base << ".vtk";
	
	// Build patches and write to file
	data_out.build_patches();
	std::ofstream output(file_name.str());
	data_out.write_vtk(output);

	// Clear objects
	data_out.clear();
}


template <>
void VelocityData<3>::output_fracture_velocity(const FractureNetwork*, double) const
{
	Assert(false, ExcNotImplemented());
}


template <>
void VelocityData<2>::output_fracture_velocity(const FractureNetwork* fractures, double width) const
{
	const unsigned int dim = 2;
	typedef typename DoFHandler<dim>::active_cell_iterator ACI;
	Vector<double> cell_values(4);
	
	unsigned int fid = 1;
	for (FractureNetwork::ConstIterator fracture = fractures->begin(); fracture != fractures->end(); ++fracture, ++fid) {
		std::ostringstream file;
		file << "output/velocity-frac" << fid << ".csv";
		std::ofstream ofs(file.str().c_str());
		ofs << "\"x\",\"y\",\"u\"" << std::endl;
		
		typename std::vector<Intersection<dim>>::const_iterator intersection = fracture->begin_intersection();
		for ( ; intersection != fracture->end_intersection(); ++intersection) {
			const Point<dim> x = fracture->vertex(intersection->index);
			const ACI cell(&(dh.get_triangulation()), intersection->cells[0].first, intersection->cells[0].second, &dh);
			const unsigned int face = intersection->faces[0];
			
			cell->get_dof_values(dof_values, cell_values);
			
			if (cell->at_boundary(face) && cell->face(face)->boundary_id() == 1)
				cell_values *= width / cell->face(face)->measure();
			
			ofs << x[0] << "," << x[1] << "," << cell_values[face]/width << std::endl;
		}
		ofs.close();
	}
}


template <int dim>
double VelocityData<dim>::scalar_product(Vector<double> v, Tensor<1,dim,double> p)
{
	AssertDimension(v.size(), dim);
	double prod = 0.0;
	for (unsigned int d=0; d<dim; ++d) {
		prod += v(d)*p[d];
	}
	return prod;
}


template <int dim>
Point<dim> VelocityData<dim>::vector_to_point(Vector<double> vec) const
{
	Point<dim> p;
	AssertDimension(vec.size(), dim);
	if (dim == 1)
		p = Point<dim>(vec(0));
	else if (dim == 2)
		p = Point<dim>(vec(0), vec(1));
	else if (dim == 3)
		p = Point<dim>(vec(0), vec(1), vec(2));
	else
		Assert(false, ExcInternalError());
	return p;
}



#endif // VELOCITY_H