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

#ifndef PARAMETER_HANDLER_FUNCTIONS
#define PARAMETER_HANDLER_FUNCTIONS


#include <deal.II/base/parameter_handler.h>


using namespace dealii;

/* Declare parameters to be stored by ParameterHandler param.
 * Also, set the default values.
 */


void declare_parameters(ParameterHandler& param)
{
	param.enter_subsection("Grid");
	{
		param.declare_entry("Dimension", 
							"2", 
							Patterns::Integer(), 
							"Problem dimension");
		param.declare_entry("Grid type",
							"Regular",
							Patterns::Anything(),
							"Grid type, either '1D', 'Regular' or 'VTK'");
		param.declare_entry("Uniform 1D grid",
							"true",
							Patterns::Bool(),
							"If true, then 1D grid is uniform. Neglected for Regular grid type");
		param.declare_entry("VTK file",
							"grid.vtk",
							Patterns::Anything(),
							"VTK file name (if VTK format)");
		param.declare_entry("Global refinement",
						    "4",
							Patterns::Integer(),
							"Number of global refinements");
		param.declare_entry("Do local refinement",
							"false",
							Patterns::Bool(),
							"If true, then do local refinement.");
		param.declare_entry("Distortion factor",
							"0.0",
							Patterns::Double(),
							"Distortion factor for vertices (to produce irregular mesh)");
	}
	param.leave_subsection();
	param.enter_subsection("Global");
	{
		param.declare_entry("Problem type",
						    "LOWPERM_REGION",
							Patterns::Anything(),
							"Problem type, see ProblemFunctions.h for a list of choices");
		param.declare_entry("Time step size",
							"0.01",
							Patterns::Double(),
							"Time step size");
		param.declare_entry("Time step size 2",
							"0.01",
							Patterns::Double(),
							"Time step size after 'Time to change dt'");
		param.declare_entry("Time to change dt",
							"5.0",
							Patterns::Double(),
							"After this time, use step size 2");
		param.declare_entry("End time",
							"5.0",
							Patterns::Double(),
							"End time for simulations");
		param.declare_entry("No quadrature points",
				            "2",
							Patterns::Double(),
							"Number of quadrature points");
	}
	param.leave_subsection();
	param.enter_subsection("Pressure solver");
	{
		param.declare_entry("Alpha",
							"1.0",
							Patterns::Double(),
							"Coefficient in front of dp/dt term");
		param.declare_entry("Dirichlet penalty",
						    "10.0",
							Patterns::Double(),
							"Dirichlet penalty coefficient (if weak boundary conditions)");
		param.declare_entry("Output file base",
							"solution_pressure",
							Patterns::Anything(),
							"Where to store results from pressure solver");
		param.declare_entry("S form",
							"1.0",
							Patterns::Double(),
							"S form = -1.0 for NIPG or OBB-DG; = 1.0 for SIPG; = 0.0 for IIPG");
		param.declare_entry("Weak BCs",
							"true",
							Patterns::Bool(),
							"Use weak Dirichlet boundary conditions or not");
		param.declare_entry("Linear solver tolerance",
							"1e-10",
							Patterns::Double(),
							"Linear solver tolerance");
		param.declare_entry("SSOR relaxation coeff",
							"1.5",
							Patterns::Double(),
							"Relaxation coefficient for SSOR preconditioner");
	}
	param.leave_subsection();
	param.enter_subsection("Postprocessing");
	{
		param.declare_entry("Method",
							"MM",
							Patterns::Anything(),
							"Either 'MM', 'GS' or 'NONE'");
		param.declare_entry("Residual tolerance",
							"1e-6",
							Patterns::Double(),
							"Tolerance when checking for local and global conservation.");
		param.declare_entry("Update neumann boundary",
							"false",
							Patterns::Bool(),
							"Perform postprocessing on Neumann boundary?");
		param.declare_entry("Update dirichlet boundary",
							"true",
							Patterns::Bool(),
							"Perform postprocessing on Dirichlet boundary?");
		param.declare_entry("Dirichlet flux recovery",
							"false",
							Patterns::Bool(),
							"Use flux recovery on Dirichlet boundary?");
		param.declare_entry("Harmonic weighting",
							"false",
							Patterns::Bool(),
							"Use harmonic weighting of flux on element edges?");
		param.declare_entry("Weighted L2-norm",
							"true",
							Patterns::Bool(),
							"Minimize correction in weighted L2-rnom?");
	}
	param.leave_subsection();
	param.enter_subsection("Transport solver");
	{
		param.declare_entry("Degree",
							"0",
							Patterns::Selection("0|1"),
							"Polynomial degree for DG space, either 0 or 1");
		param.declare_entry("Solver tolerance",
							"1e-10",
							Patterns::Double(),
							"Tolerance for linear solver");
		param.declare_entry("Do transport",
							"true",
							Patterns::Bool(),
							"Run transport solver?");
		param.declare_entry("Output file base",
							"solution_transport",
							Patterns::Anything(),
							"Where to store results from transport solver");
		param.declare_entry("Inflow concentration",
							"1.0",
							Patterns::Double(),
							"Inflow concentration on inflow boundary and from source");
		param.declare_entry("Use exact velocity",
							"false",
							Patterns::Bool(),
							"Use exact velocity in transport solver");
		param.declare_entry("Stride",
							"1",
							Patterns::Integer(),
							"Output every stride'th time step");
	}
	param.leave_subsection();
}



#endif // PARAMETER_HANDLER_FUNCTIONS