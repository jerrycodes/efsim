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

#include "../source/DirichletFluxRecovery.h"
#include "../source/EllipticPressureSolver.h"
#include "../source/HelpFunctions.h"

using namespace dealii;

/*
 * Test DirichletFluxRecovery
 */

int main (int argc, char *argv[])
{
	deallog.depth_console(0);
	Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

	const ProblemType pt = ProblemType::ANALYTIC;

	const int dim = 2;
	Triangulation<dim> triangulation;
	make_grid(triangulation, pt, 3, true, 0.3);
	//make_grid_1D(triangulation, 0.5);

	const FE_Q<dim> fe_pressure(1);
	DoFHandler<dim> dh_pressure(triangulation);

	ProblemFunctionsFlow<dim> flow_fun(pt);
	RockProperties<dim> rock(triangulation);
	rock.initialize(pt);

	EllipticPressureSolver<dim> elliptic_solver(triangulation, fe_pressure, dh_pressure, rock, flow_fun);
	elliptic_solver.set_parameters(10.0, "solution_pressure", 2, false);
	elliptic_solver.run();

	DirichletFluxRecovery<dim> dirichlet_flux_recovery(dh_pressure);
	dirichlet_flux_recovery.set_parameters(2);
	dirichlet_flux_recovery.setup_and_assemble_matrix();
	dirichlet_flux_recovery.construct_rhs_steady(elliptic_solver.get_laplace_matrix(),
												 elliptic_solver.get_rhs_no_bcs(),
												 elliptic_solver.get_pressure_solution());
	dirichlet_flux_recovery.solve();
	bool gc = dirichlet_flux_recovery.globally_conservative(fe_pressure, new Neumann<dim,pt>(), new RightHandSide<dim,pt>());

	if (! gc) {
		std::cerr << "ERROR: Flux is not globally conservative!\n";
		return 1;
	}

	return 0;
}
