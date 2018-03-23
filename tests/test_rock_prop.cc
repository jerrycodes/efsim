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

#include <deal.II/base/logstream.h>

#include "../source/RockProperties.h"
#include "../source/HelpFunctions.h"
#include "../source/ProblemFunctions.h"


using namespace dealii;


static unsigned int test_nr = 1;

template <class Type>
void compare(Type val1, Type val2) 
{
	if (val1 != val2) {
		std::cout << "Test nr. " << test_nr << " failed!" << std::endl
		          << "  val1 = " << val1 << std::endl
		          << "  val2 = " << val2 << std::endl;
		exit(1);
	}
}


int main (int, char**)
{
	deallog.depth_console(0);
	const int dim = 2;
	const ProblemType pt = ProblemType::LOWPERM_REGION;
	
	// Test RockProperties
	{
		Triangulation<dim> triangulation;
		make_grid(triangulation, pt, 3);
		
		RockProperties<dim> rock(triangulation);
		const double value = 0.3;
		
		// TEST
		{
			rock.initialize(value, value);
			typename Triangulation<dim>::active_cell_iterator
			cell = triangulation.begin_active(),
			endc = triangulation.end();
			for ( ; cell != endc; ++cell) {
				compare(rock.get_poro(cell), value);
				compare(rock.get_perm(cell)[0][0], value);
				compare(rock.get_perm(cell)[1][1], value);
				compare(rock.get_perm(cell)[0][1], 0.0);
				compare(rock.get_perm(cell)[1][0], 0.0);
			}
			rock.clear();
			++test_nr;
		}
		
		// TEST
		{
			std::vector<double> data_vec(triangulation.n_active_cells(), value);
			data_vec[3] = 2*value;
			rock.initialize(value, data_vec);
			typename Triangulation<dim>::active_cell_iterator
			cell = triangulation.begin_active(),
			endc = triangulation.end();
			unsigned int count = 0;
			for ( ; cell != endc; ++cell, ++count) {
				compare(rock.get_poro(cell), value);
				compare(rock.get_perm(cell)[0][0], data_vec[count]);
				compare(rock.get_perm(cell)[1][1], data_vec[count]);
				compare(rock.get_perm(cell)[0][1], 0.0);
				compare(rock.get_perm(cell)[1][0], 0.0);
			}
			rock.clear();
			++test_nr;
		}
		
		// TEST
		{
			std::vector<double> poro_vec(triangulation.n_active_cells(), value);
			poro_vec[3] = 2*value;
			Tensor<2,dim> perm;
			perm[0][0] = 2.0; perm[0][1] = 1.5; perm[1][1] = 1.0;
			std::vector<Tensor<2,dim>> perm_vec(triangulation.n_active_cells(), perm);
			perm_vec[3] *= 2;
			perm_vec[2][0][1] = 0.5;
			rock.initialize(poro_vec, perm_vec);
			typename Triangulation<dim>::active_cell_iterator
			cell = triangulation.begin_active(),
			endc = triangulation.end();
			unsigned int count = 0;
			for ( ; cell != endc; ++cell, ++count) {
				compare(rock.get_poro(cell), poro_vec[count]);
				compare(rock.get_perm(cell), perm_vec[count]);
			}
			rock.clear();
			++test_nr;
		}
		
		// TEST
		{
			std::vector<double> perm_x(triangulation.n_active_cells(), value);
			perm_x[3] = value/2.0;
			std::vector<double> perm_y(triangulation.n_active_cells(), 2*value);
			
			std::vector<std::vector<double>> perm;
			perm.push_back(perm_x);
			perm.push_back(perm_y);

			rock.initialize(value, perm);
			typename Triangulation<dim>::active_cell_iterator
			cell = triangulation.begin_active(),
			endc = triangulation.end();
			unsigned int count = 0;
			for ( ; cell != endc; ++cell, ++count) {
				compare(rock.get_poro(cell), value);
				compare(rock.get_perm(cell)[0][0], perm_x[count]);
				compare(rock.get_perm(cell)[1][1], perm_y[count]);
				compare(rock.get_perm(cell)[0][1], 0.0);
				compare(rock.get_perm(cell)[1][0], 0.0);
			}
			rock.clear();
			++test_nr;
		}
		
		// TEST
		{
			rock.initialize(pt);
			ProblemFunctionsRock<dim> fun(pt);
			typename Triangulation<dim>::active_cell_iterator
			cell = triangulation.begin_active(),
			endc = triangulation.end();
			for ( ; cell != endc; ++cell) {
				compare(rock.get_poro(cell), fun.porosity->value(cell->center()));
				compare(rock.get_perm(cell), fun.permeability->value(cell->center()));
			}
			rock.clear();
			++test_nr;
		}
	}
	
	// Test refinement with RockProperties
	{
		// TEST
		{
			Triangulation<dim> triangulation;
			RockProperties<dim> rock(triangulation);
			make_grid(triangulation, pt, 1);
			std::vector<double> poro(4,0.1);
			poro[1] = 0.2;
			poro[2] = 0.3;
			poro[3] = 0.4;
			
			std::vector<double> perm(4,1.0);
			perm[0] = 2.0;
			rock.initialize(poro, perm);
			rock.print();
			
			// First refine the two first cells
			typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
			cell->set_refine_flag();
			++cell;
			cell->set_refine_flag();
			triangulation.execute_coarsening_and_refinement();
			rock.execute_coarsening_and_refinement();
			rock.print();
			
			std::vector<double> poro_intermediate_grid(10,0.1);
			poro_intermediate_grid[0] = 0.3;
			poro_intermediate_grid[1] = 0.4;
			poro_intermediate_grid[6] = 0.2;
			poro_intermediate_grid[7] = 0.2;
			poro_intermediate_grid[8] = 0.2;
			poro_intermediate_grid[9] = 0.2;
			
			unsigned int count = 0;
			for ( cell = triangulation.begin_active(); cell != triangulation.end(); ++cell, ++count) {
				compare(rock.get_poro(cell), poro_intermediate_grid[count]);
			}
			
			// Then refine bottom left and coarsen bottom right
			for ( cell = triangulation.begin_active(); cell != triangulation.end(); ++cell) {
				if (cell->center().norm() < 0.25)
					cell->set_refine_flag();
				else if (cell->center().distance(Point<2>(1.0,0.0)) < 0.55)
					cell->set_coarsen_flag();
			}
			rock.prepare_coarsening();
			triangulation.execute_coarsening_and_refinement();
			rock.execute_coarsening_and_refinement();
			rock.print();
			
			std::vector<double> poro_end_grid(10,0.1);
			poro_end_grid[0] = 0.2;
			poro_end_grid[1] = 0.3;
			poro_end_grid[2] = 0.4;
			
			count = 0;
			for ( cell = triangulation.begin_active(); cell != triangulation.end(); ++cell, ++count) {
				compare(rock.get_poro(cell), poro_end_grid[count]);
			}
			
			rock.clear();
			++test_nr;
		}
		
		// TEST
		{
			Triangulation<dim> triangulation;
			RockProperties<dim> rock(triangulation);
			make_grid(triangulation, pt, 1);
			std::vector<double> poro(4,0.1);
			poro[1] = 0.2;
			poro[2] = 0.3;
			poro[3] = 0.4;
			
			std::vector<double> perm(4,1.0);
			perm[0] = 2.0;
			rock.initialize(poro, perm);
			rock.print();

			// Merge all cells into one.
			typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
			for ( ; cell != triangulation.end(); ++cell) {
				cell->set_coarsen_flag();
			}
			// rock.check_refinement_and_coarsening() should return true
			if ( ! rock.prepare_coarsening()) {
				exit(1);
			}
			triangulation.execute_coarsening_and_refinement();
			rock.execute_coarsening_and_refinement();
			rock.print();
			
			rock.clear();
			++test_nr;
		}
	}
	
	return 0;
}
