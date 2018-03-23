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

#include <deal.II/base/parameter_handler.h>

#include "../source/CoupledFlowTransport.h"
#include "../source/ParameterHandlerFunctions.h"

using namespace dealii;


unsigned int read_parameters(int argc, char *argv[], ParameterHandler& param);


/* Main program.
 * Binary takes a parameter file as input and create and run
 * the coupled flow and transport problem with optinal postprocessing.
 */
int main (int argc, char *argv[])
{
	deallog.depth_console (5);
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

	ParameterHandler param;
	unsigned int dim = read_parameters(argc, argv, param);

	switch (dim)
	{
		case 2:
			{
				CoupledFlowTransport<2> coupled_problem;
				coupled_problem.read_parameters(param);
				coupled_problem.run();
			}
			break;
		case 3:
			{
				CoupledFlowTransport<3> coupled_problem;
				coupled_problem.read_parameters(param);
				coupled_problem.run();
			}
			break;
		default:
			ExcInternalError();
	}

	return 0;
}


unsigned int read_parameters(int argc, char *argv[], ParameterHandler& param)
{
	// Declare parameters and read input file
	declare_parameters(param);
	if (argc < 2)
		std::cout << "Warning: No input given. Using default setup."
				  << std::endl << std::endl;
	else {
		char* parameter_file = argv[1];
		if ( ! param.read_input(parameter_file, true) ) {
			std::cout << "  -> Using default setup." << std::endl << std::endl;
		}
		if (argc > 2)
			std::cout << "Warning: More than one input argument given. Ignoring all but the first."
					  << std::endl << std::endl;
	}
	
	// Read and return dimension
	param.enter_subsection("Grid");
	unsigned int dim = param.get_integer("Dimension");
	param.leave_subsection();
	
	if (dim < 2 || dim > 3) {
		std::cout << "Error! Dimension should be 2 or 3!\n";
		exit(1);
	}
	
	return dim;
}
