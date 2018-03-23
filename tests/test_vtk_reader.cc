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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>

#include "../source/VTKReader.h"
#include "../source/HelpFunctions.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace dealii;

/*
 * Test VTKReader
 */


// Simple routine for solving laplace
template <int dim>
bool solve_laplace(Triangulation<dim>& tria)
{
	FE_Q<dim> fe(1);
	DoFHandler<dim> dh(tria);
	dh.distribute_dofs(fe);

	SparseMatrix<double> laplace_matrix;
	Vector<double> system_rhs;
	Vector<double> solution;
	SparsityPattern sparsity;
	ConstraintMatrix constraints;

	DoFTools::make_hanging_node_constraints(dh, constraints);
	constraints.close();
	DynamicSparsityPattern c_sparsity(dh.n_dofs());
	DoFTools::make_sparsity_pattern(dh, c_sparsity, constraints);
	sparsity.copy_from(c_sparsity);

	Assert(!sparsity.empty(), ExcMessage("Sparsity pattern is empty.\n"));

	laplace_matrix.reinit(sparsity);
	system_rhs.reinit(dh.n_dofs());
	solution.reinit(dh.n_dofs());

	laplace_matrix = 0;
	std::cout << "  Assemble laplace matrix ...\n";
	MatrixCreator::create_laplace_matrix(dh,
										 QGauss<dim>(2),
										 laplace_matrix,
										 (const Function<dim>*)0,
									     constraints);
	std::cout << "  Assemble rhs ...\n";
	VectorTools::create_right_hand_side(dh, QGauss<dim>(2), ConstantFunction<dim>(2*dim), system_rhs);
	
	std::cout << "  Assemble boundary values ...\n";
	std::map<types::global_dof_index,double> boundary_values;
	VectorTools::interpolate_boundary_values(dh,
											 0,
											 ZeroFunction<dim>(),
											 boundary_values);
	MatrixTools::apply_boundary_values(boundary_values,
									   laplace_matrix,
									   solution,
									   system_rhs);
	
	SolverControl solver_control (dh.n_dofs(), 1e-12);			
	SolverCG<> solver(solver_control);
	solver.solve(laplace_matrix, solution, system_rhs, PreconditionIdentity());
	
	// Output results
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dh);
	data_out.add_data_vector(solution, "pressure");
	data_out.build_patches();
	std::ofstream output("laplace_solution.vtk");
	data_out.write_vtk (output);

	laplace_matrix.clear();
	dh.clear();
	
	const double true_solution_midpoint = 0.5625;
	if ( std::abs(solution[7] - true_solution_midpoint) < 1e-10 ) return 1;
	else return 0;
}


int main(int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
	deallog.depth_console (0);
	
	// Check number of input variables
	if (argc != 2) {
		std::cout << "Error: No input file given." << std::endl;
		std::cout << "Usage: ./read_cpgrid gridfile.vtk" << std::endl;
		exit(1);
	}

	// Check if input file is readable
	const char* vtkfilename = argv[1];
	std::ifstream vtkfile(vtkfilename);
	if ( ! vtkfile.is_open()) {
		std::cout << "Error: Not able to read input file." << std::endl;
		std::cout << "Usage: ./read_cpgrid gridfile.vtk" << std::endl;
		exit(1);
	}

	const int dim = 3;
	
	// Triangulation to store cpgrid
	Triangulation<dim> tria_cpgrid;

	VTKReader<dim> grid_in;
	grid_in.attach_triangulation(tria_cpgrid);
	grid_in.read_vtk(vtkfile);
	vtkfile.close();

	std::cout << "Grid info:" << std::endl;
	std::cout << "  Number of cells:    " << tria_cpgrid.n_active_cells() << std::endl;
	std::cout << "  Number of faces:    " << tria_cpgrid.n_active_faces() << std::endl;
	std::cout << "  Number of vertices: " << tria_cpgrid.n_vertices() << std::endl;
	
	// Solve Laplace problem
	std::cout << "Solve Laplace problem" << std::endl;
	if (solve_laplace<dim>(tria_cpgrid)) 
		std::cout << "Laplace problem solved correctly!" << std::endl;
	else {
		std::cout << "Did not solve Laplace problem correctly!" << std::endl;
		return 1;
	}
	
	// Check poro and perm (holds only for 8cellmodel.vtk)
	std::vector<double> true_poro(8, 0.2);
	std::vector<double> true_perm(8, 1.0);
	for (unsigned int i=4; i<8; ++i)
		true_perm[i] = 10.0;

	if ( grid_in.get_poro() == true_poro )
		std::cout << "Porosity read correctly!" << std::endl;
	else {
		std::cout << "Porosity is not read correctly" << std::endl;
		return 1;
	}
	if ( grid_in.get_permx() == true_perm )
		std::cout << "Permeability read correctly!" << std::endl;
	else {
		std::cout << "Permeability is not read correctly" << std::endl;
		return 1;
	}

	return 0;
}
