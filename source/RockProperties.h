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

#ifndef ROCK_PROPERTIES_H
#define ROCK_PROPERTIES_H

#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <boost/concept_check.hpp>

#include "ProblemFunctions.h"


using namespace dealii;


// Struct to store material data (poro/perm)
template<int dim>
struct MaterialData
{
	MaterialData(double poro_in, Tensor<2,dim> perm_in) { poro = poro_in; perm = perm_in; }
	double poro;
	Tensor<2,dim> perm;
};


// Rock interface to store permeability and porosity.
// All data is mapped from the user index of a cell.
// Tried using material id, but this is limited to the range [0,255]
template <int dim>
class RockProperties
{
public:
	// Constructor
	RockProperties(Triangulation<dim> &tria);
	
	typedef TriaActiveIterator<CellAccessor<dim>> CellPointer;
	
	// Initialize properties either from various poro/perm data or from ProblemType
	// PoroType = {double, std::vector<double>, Function<dim>*}
	// PermType = {double, Tensor<2,dim>, std::vector<double>, std::vector<Tensor<2,dim>>,
	//             std::vector<std::vector<double>>, TensorFunction<2,dim>*}
	template<class PoroType, class PermType>
	void initialize(PoroType poro, PermType perm);
	void initialize(ProblemType pt);
	
	// Call this prior to refinement
	bool prepare_coarsening();
	void execute_coarsening_and_refinement();
	
	bool is_isotropic() const;
	
	// Clear material data
	void clear();
	
	// Get functions
	Tensor<2,dim> get_perm(CellPointer cell) const;
	double        get_poro(CellPointer cell) const;
	std::vector<double>        get_poro() const;
	std::vector<Tensor<2,dim>> get_perm() const;
	std::vector<double>        get_perm_comp(unsigned int comp) const;
	
	// Print data cell-wise
	void print(std::ostream& os = std::cout) const;
	
private:
	const SmartPointer<Triangulation<dim>> triangulation;
	unsigned int n_cells;
	
	bool isotropic;
	
	// Minimum grid level when coarsening grid
	// We can't coarsen cells whose level are smaller than this
	unsigned int min_grid_level;
	
	// Map user index to poro and perm. User index should never be zero (this means not set)
	std::map<unsigned int, MaterialData<dim>> map_idx_to_data;
	typedef std::pair<unsigned int, MaterialData<dim>> Pair;
	
	// dim x dim identity matrix
	Tensor<2,dim> eye;
	
	void initialize_common();
	
	// Set poro and perm
	template<class PoroType, class PermType>
	void set_poro_and_perm(PoroType poro, PermType perm);
	// Specializations
	void set_poro_and_perm(double, double);
	void set_poro_and_perm(double, Tensor<2,dim>);
	void set_poro_and_perm(std::vector<double>, std::vector<Tensor<2,dim>>);
	void set_poro_and_perm(double, double, double, double);
	
	template <class PoroType>
	std::vector<double> generate_poro_vec(PoroType poro);
	template <class PermType>
	std::vector<Tensor<2,dim>> generate_perm_vec(PermType perm);
	// Specializations
	std::vector<double> generate_poro_vec(double);
	std::vector<double> generate_poro_vec(std::vector<double>);
	std::vector<double> generate_poro_vec(Function<dim>*);
	std::vector<Tensor<2,dim>> generate_perm_vec(double);
	std::vector<Tensor<2,dim>> generate_perm_vec(Tensor<2,dim>);
	std::vector<Tensor<2,dim>> generate_perm_vec(std::vector<double>);
	std::vector<Tensor<2,dim>> generate_perm_vec(std::vector<std::vector<double>>);
	std::vector<Tensor<2,dim>> generate_perm_vec(std::vector<Tensor<2,dim>>);
	std::vector<Tensor<2,dim>> generate_perm_vec(TensorFunction<2,dim>*);
};


template <int dim>
RockProperties<dim>::RockProperties(Triangulation<dim> &tria)
: triangulation(&tria)
{}


template <int dim>
void RockProperties<dim>::initialize_common()
{
	Assert(map_idx_to_data.empty(), ExcMessage("Not allowed to reset poro or perm. Use clear function."));
	min_grid_level = triangulation->n_levels() -1;
	for (unsigned int d=0; d<dim; ++d)
		eye[d][d] = 1.0;
	n_cells = triangulation->n_active_cells();
	Assert(n_cells > 0, ExcMessage("Triangulation is empty."));
}


template <int dim>
void RockProperties<dim>::initialize(ProblemType pt)
{
	initialize_common();
	if (pt == SIMPLE_FRAC_RESOLVED) {
		set_poro_and_perm(1.0, 1.0, 1.0, 1000.0);
		Assert(map_idx_to_data.count(0) == 0, ExcInternalError());
		return;
	}
	ProblemFunctionsRock<dim> funs(pt);
	set_poro_and_perm(funs.porosity, funs.permeability);
	// Key 0 (user index) is not allowed
	Assert(map_idx_to_data.count(0) == 0, ExcInternalError());
}


template <int dim>
template<class PoroType, class PermType>
void RockProperties<dim>::initialize(PoroType poro, PermType perm)
{
	initialize_common();
	set_poro_and_perm(poro, perm);
	// Key 0 (user index) is not allowed
	Assert(map_idx_to_data.count(0) == 0, ExcInternalError());
}


template <int dim>
void RockProperties<dim>::clear()
{
	map_idx_to_data.clear();
	triangulation->clear_user_data();
}


// Check if we try to coarsen below original grid.
// This may lead to merging of cells that have different user index.
// Clear coarsen flags for such cells and return if some flags are cleared.
template <int dim>
bool RockProperties<dim>::prepare_coarsening()
{
	bool cleared_flags = false;
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation->begin_active(min_grid_level),
	endc = triangulation->end_active(min_grid_level);
	for ( ; cell != endc; ++cell) {
		if (cell->coarsen_flag_set()) {
			cleared_flags = true;
			cell->clear_coarsen_flag();
		}
	}
	return cleared_flags;
}


// User index needs to be inherited
// Call this after triangulation->execute_refinement_and_coarsening()
template <int dim>
void RockProperties<dim>::execute_coarsening_and_refinement()
{
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation->begin_active(),
	endc = triangulation->end();
	for ( ; cell != endc; ++cell) {
		if ( !(cell->user_index()) )
			cell->set_user_index(cell->parent()->user_index());
	}
}


template <int dim>
bool RockProperties<dim>::is_isotropic() const
{
	return isotropic;
}


template <int dim>
Tensor<2,dim> RockProperties<dim>::get_perm(RockProperties<dim>::CellPointer cell) const
{
	Assert(cell->user_index(), ExcMessage("User index not set. Have you forgotten to initialize or call execute_coarsening_and_refinement()?"));
	Assert(cell->user_index() > 0, ExcInternalError());
	return map_idx_to_data.at(cell->user_index()).perm;
}


template <int dim>
std::vector<Tensor<2,dim>> RockProperties<dim>::get_perm() const
{
	std::vector<Tensor<2,dim>> perm_comp;
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation->begin_active(),
	endc = triangulation->end();
	for (; cell!=endc; ++cell) {
		perm_comp.push_back(get_perm(cell));
	}
	return perm_comp;
}


template <int dim>
std::vector<double> RockProperties<dim>::get_perm_comp(unsigned int comp) const
{
	std::vector<double> perm_comp;
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation->begin_active(),
	endc = triangulation->end();
	for (; cell!=endc; ++cell) {
		perm_comp.push_back(get_perm(cell)[comp][comp]);
	}
	return perm_comp;
}


template <int dim>
double RockProperties<dim>::get_poro(RockProperties<dim>::CellPointer cell) const
{
	Assert(cell->user_index(), ExcMessage("User index not set. Have you forgotten to initialize or call execute_coarsening_and_refinement()?"));
	Assert(cell->user_index() > 0, ExcInternalError());
	return map_idx_to_data.at(cell->user_index()).poro;
}


template <int dim>
std::vector<double> RockProperties<dim>::get_poro() const
{
	std::vector<double> poro;
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation->begin_active(),
	endc = triangulation->end();
	for (; cell!=endc; ++cell) {
		poro.push_back(get_poro(cell));
	}
	return poro;
}


// Take constant poro and perm and set all user indices to 0
template <int dim>
void RockProperties<dim>::set_poro_and_perm(double poro, Tensor<2,dim> perm)
{
	isotropic = false;
	const unsigned int idx = 1;
	
	// Set user index of all cells to 0
	typename Triangulation<dim>::cell_iterator
	cell = triangulation->begin(0),
	endc = triangulation->end(0);
	for ( ; cell != endc; ++cell)
		cell->recursively_set_user_index(idx);
	map_idx_to_data.insert(Pair(idx, MaterialData<dim>(poro,perm)));
}


// Same as above, but with isotropic permeability
template <int dim>
void RockProperties<dim>::set_poro_and_perm(double poro, double perm)
{
	isotropic = true;
	set_poro_and_perm(poro, eye*perm);
}


// Take poro and perm as vectors with entries for each cell, and set user index to a unique value for every cell.
template <int dim>
void RockProperties<dim>::set_poro_and_perm(std::vector<double> poro, std::vector<Tensor<2,dim>> perm)
{
	AssertDimension(n_cells, poro.size());
	AssertDimension(n_cells, perm.size());
	
	isotropic = false;
	
	unsigned int ci = 0;
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation->begin_active(),
	endc = triangulation->end();
	for ( ; cell != endc; ++cell, ++ci) {
		const unsigned int idx_cell = ci +1;
		map_idx_to_data.insert(Pair(idx_cell, MaterialData<dim>(poro[ci],perm[ci])));
		cell->set_user_index(idx_cell);
	}
}


template <int dim>
void RockProperties<dim>::set_poro_and_perm(double poro_matrix, double poro_frac, double perm_matrix, double perm_frac)
{
	isotropic = true;
	const unsigned int idx_matrix = 1;
	const unsigned int idx_frac   = 2;
	
	unsigned int max_level = triangulation->n_levels() - 1;
	
	{
		typename Triangulation<dim>::cell_iterator
		cell = triangulation->begin(0),
		endc = triangulation->end(0);
		for ( ; cell != endc; ++cell)
			cell->recursively_set_user_index(idx_matrix);
	}
	
	{
		typename Triangulation<dim>::active_cell_iterator
		cell = triangulation->begin(max_level),
		endc = triangulation->end(max_level);
		for ( ; cell != endc; ++cell)
			cell->set_user_index(idx_frac);
	}
	
	map_idx_to_data.insert(Pair(idx_matrix, MaterialData<dim>(poro_matrix,eye*perm_matrix)));
	map_idx_to_data.insert(Pair(idx_frac,   MaterialData<dim>(poro_frac,  eye*perm_frac)));
}


// Take general poro and perm types, put them into vectors, and call specialized function
template <int dim>
template <class PoroType, class PermType>
void RockProperties<dim>::set_poro_and_perm(PoroType poro, PermType perm)
{
	// isotropic is set in generate_perm_vec(perm)
	std::vector<double>        poro_per_cell = generate_poro_vec(poro);
	std::vector<Tensor<2,dim>> perm_per_cell = generate_perm_vec(perm);
	
	set_poro_and_perm(poro_per_cell, perm_per_cell);
}


// Functions that take general poro/perm data and put them into vectors of sixe n_cells
template <int dim>
std::vector<double> RockProperties<dim>::generate_poro_vec(double poro_const)
{
	return std::vector<double>(n_cells, poro_const);
}

template <int dim>
std::vector<double> RockProperties<dim>::generate_poro_vec(std::vector<double> poro)
{
	return poro;
}

template <int dim>
std::vector<double> RockProperties<dim>::generate_poro_vec(Function<dim>* fun)
{
	std::vector<double> poro;
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation->begin_active(),
	endc = triangulation->end();
	for ( ; cell != endc; ++cell) {
		poro.push_back(fun->value(cell->center()));
	}
	return poro;
}

template <int dim>
std::vector<Tensor<2,dim>> RockProperties<dim>::generate_perm_vec(double perm_const)
{
	isotropic = true;
	return std::vector<Tensor<2,dim>>(n_cells, eye*perm_const);
}

template <int dim>
std::vector<Tensor<2,dim>> RockProperties<dim>::generate_perm_vec(Tensor<2,dim> perm_const)
{
	isotropic = false;
	return std::vector<Tensor<2,dim>>(n_cells, perm_const);
}

template <int dim>
std::vector<Tensor<2,dim>> RockProperties<dim>::generate_perm_vec(TensorFunction<2,dim>* fun)
{
	// TODO: Check if perm is really anisotropic
	isotropic = false;

	std::vector<Tensor<2,dim>> perm;
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation->begin_active(),
	endc = triangulation->end();
	for ( ; cell != endc; ++cell) {
		perm.push_back(fun->value(cell->center()));
	}
	return perm;
}

template <int dim>
std::vector<Tensor<2,dim>> RockProperties<dim>::generate_perm_vec(std::vector<Tensor<2,dim>> perm)
{
	isotropic = false;
	return perm;
}

template <int dim>
std::vector<Tensor<2,dim>> RockProperties<dim>::generate_perm_vec(std::vector<double> perm)
{
	isotropic = true;
	std::vector<Tensor<2,dim>> perm_tensor;
	for (std::vector<double>::iterator it = perm.begin(); it != perm.end(); ++it)
		perm_tensor.push_back(*it*eye);
	return perm_tensor;
}

template <int dim>
std::vector<Tensor<2,dim>> RockProperties<dim>::generate_perm_vec(std::vector<std::vector<double>> perm)
{
	const unsigned int n_comp = perm.size();
	Assert(n_comp > 0 && n_comp < dim+1, ExcMessage("Nr of components should be positive and smaller than dim"));
	if (n_comp == 1) {
		isotropic = true;
		return generate_perm_vec(perm[0]);
	}
	isotropic = false;
	// perm should have size dim, and each entry should have size n_cells
	AssertVectorVectorDimension(perm, dim, n_cells);
	std::vector<Tensor<2,dim>> perm_tensor;
	for (unsigned int ci=0; ci<n_cells; ++ci) {
		Tensor<2,dim> perm_cell;
		for (unsigned int d=0; d<dim; ++d)
			perm_cell[d][d] = perm[d][ci];
		perm_tensor.push_back(perm_cell);
	}
	return perm_tensor;
}


template <int dim>
void RockProperties<dim>::print(ostream& os) const
{
	int w = 10;
	os << "Rock parameters per cell:\n\n";
	os << std::right << std::setw(w/2) << "Cell"
	   << std::right << std::setw(w) << "CellIndex"
	   << std::left << std::setw(w) << "  Poro"
	   << std::left << std::setw(w) << "  Perm" << std::endl;
	unsigned int ci = 0;
	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation->begin_active(),
	endc = triangulation->end();
	for ( ; cell != endc; ++cell, ++ci) {
		os << std::right << std::setw(w/2) << ci
		   << std::right << std::setw(w) << cell->user_index()
		   << std::right << std::setw(w) << std::setprecision(5) << std::fixed << get_poro(cell)
		   << std::right << std::setw(w) << std::setprecision(2) << std::scientific << get_perm(cell)  << std::endl;
	}
}


#endif // ROCK_PROPERTIES_H