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

#ifndef VTK_READER_H
#define VTK_READER_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_reordering.h>
#include <boost/concept_check.hpp>

#include "HelpFunctions.h"

using namespace dealii;


/*
 * Class with routines to read vtk format (Simple Legacy Format)
 * Based on dealII::GridIn, but with appropriate modifications"
 */


template <int dim>
class VTKReader
{
public:
	VTKReader() {}
	void attach_triangulation(Triangulation<dim> &tria) { triangulation = &tria; }

	void read_vtk(std::istream &in);

	std::vector<double> get_poro()  const { Assert(have_poro,    ExcMessage("Poro is not available."));  return poro; }
	std::vector<double> get_permx() const { Assert(have_perm[0], ExcMessage("Permx is not available.")); return perm[0]; }
	std::vector<double> get_permy() const { Assert(have_perm[1], ExcMessage("Permy is not available.")); return perm[1]; }
	std::vector<double> get_permz() const { Assert(have_perm[2], ExcMessage("Permz is not available.")); return perm[2]; }
	std::vector<std::vector<double>> get_perm() const { Assert(have_perm[0], ExcMessage("Perm is not available.")); return perm; }

private:
	SmartPointer<Triangulation<dim>> triangulation;

	bool structured_grid;

	std::vector<double> poro;
	std::vector<std::vector<double>> perm;

	bool have_poro = false;
	std::vector<bool> have_perm = std::vector<bool>(3, false);

	unsigned int hexahedron_vertex_map[8] = {0, 1, 3, 2, 4, 5, 7, 6};
};


template <int dim>
void VTKReader<dim>::read_vtk(istream& in)
{
	Assert((dim == 2)||(dim == 3), ExcNotImplemented());
	std::string line;

	// verify that the first, third and fourth lines match
	// expectations. the second line of the file may essentially be
	// anything the author of the file chose to identify what's in
	// there, so we just ensure that we can read it
	{
		std::string text[4];
		text[0] = "# vtk DataFile Version 3.0";
		text[1] = "****";
		text[2] = "ASCII";
		text[3] = "DATASET UNSTRUCTURED_GRID";

		for (unsigned int i = 0; i < 3; ++i)
		{
			getline(in,line);
			if (i != 1)
			AssertThrow (line.compare(text[i]) == 0,
                         ExcMessage(std::string("While reading VTK file, failed to find a header line with text <") +
						 text[i] + ">"));
		}

		getline(in,line);
		if ( line.compare("DATASET UNSTRUCTURED_GRID") == 0 )
			structured_grid = false;
		else if ( line.compare("DATASET STRUCTURED_GRID") == 0 )
			structured_grid = true;
		else
			Assert(false, ExcMessage("While reading VTK file, failed to find a header line with text <DATASET [UN]STRUCTURED_GRID>"));
	}


	///////////////////Declaring storage and mappings//////////////////

	Point<dim> N;
	std::vector< Point<dim> > vertices;//vector of vertices
	std::vector< CellData<dim> > cells;//vector of cells
	SubCellData subcelldata;//subcell data that includes bounds and material IDs.
	std::map<int, int> vertex_indices; // # vert in unv (key) ---> # vert in deal.II (value)
	std::map<int, int> cell_indices; // # cell in unv (key) ---> # cell in deal.II (value)
	std::map<int, int> quad_indices; // # quad in unv (key) ---> # quad in deal.II (value)
	std::map<int, int> line_indices; // # line in unv(key) ---> # line in deal.II (value)

	unsigned int no_vertices, no_quads=0, no_lines=0;

	std::string keyword;

	if (structured_grid) {
		in >> keyword;
		if (keyword == "DIMENSIONS") {
			for (unsigned int d=0; d<dim; ++d)
				in >> N[d];
		}
		else
			AssertThrow (false, ExcMessage ("While reading VTK file, failed to find DIMENSIONS section"));
	}

	in >> keyword;

	//////////////////Processing the POINTS section///////////////

	if (keyword == "POINTS")
	{
		in>>no_vertices;// taking the no. of vertices
		in.ignore(256, '\n');//ignoring the number beside the total no. of points.
		for (unsigned int count = 0; count < no_vertices; count++) //loop to read three values till the no . vertices is satisfied
		{
			// VTK format always specifies vertex coordinates with 3 components
			Point<3> x;
			in >> x(0) >> x(1) >> x(2);
			vertices.push_back(Point<dim>());
			for (unsigned int d=0; d<dim; ++d)
				vertices.back()(d) = x(d);
			vertex_indices[count] = count;
		}
	}
	else
		AssertThrow (false, ExcMessage ("While reading VTK file, failed to find POINTS section"));

	//////////////////ignoring space between points and cells sections////////////////////
	std::string checkline;
	int no;
	in.ignore(256, '\n');//this move pointer to the next line ignoring unwanted no.
	no = in.tellg();
	getline(in,checkline);
	if (checkline.compare("") != 0)
	{
		in.seekg(no);
	}
	in >> keyword;
	unsigned int total_cells, no_cells = 0, type;// declaring counters, refer to the order of declaring variables for an idea of what is what!

	///////////////////Processing the CELLS section that contains cells(cells) and bound_quads(subcelldata)///////////////////////

	if ( !structured_grid ) {
		if (keyword == "CELLS")
		{
			in>>total_cells;
			in.ignore(256,'\n');

			if (dim == 3)
			{
				for (unsigned int count = 0; count < total_cells; count++)
				{
					in>>type;
					if (type == 8)
					{
						cells.push_back(CellData<dim>());
						for (unsigned int j = 0; j < type; j++) //loop to feed data
						{
							in >> cells.back().vertices[hexahedron_vertex_map[j]];
						}

						cells.back().material_id = 0;
						for (unsigned int j = 0; j < type; j++) //loop to feed the data of the vertices to the cell
						{
							cells.back().vertices[j] = vertex_indices[cells.back().vertices[j]];
						}
						cell_indices[count] = count;
						no_cells++;
					}
					else if ( type == 4)
					{
						subcelldata.boundary_quads.push_back(CellData<2>());
						for (unsigned int j = 0; j < type; j++) //loop to feed the data to the boundary
						{
							in >> subcelldata.boundary_quads.back().vertices[j];
						}
						subcelldata.boundary_quads.back().material_id = 0;
						for (unsigned int j = 0; j < type; j++)
						{
							subcelldata.boundary_quads.back().vertices[j] = vertex_indices[subcelldata.boundary_quads.back().vertices[j]];
						}
						quad_indices[no_quads] = no_quads + 1;
						no_quads++;
					}
					else
						AssertThrow (false, ExcMessage ("While reading VTK file, unknown file type encountered"));
				}
			}
			else if (dim == 2)
			{
				for (unsigned int count = 0; count < total_cells; count++)
				{
					in>>type;
					if (type == 4)
					{
						cells.push_back(CellData<dim>());
						for (unsigned int j = 0; j < type; j++) //loop to feed data
							in >> cells.back().vertices[j];
						cells.back().material_id = 0;
						for (unsigned int j = 0; j < type; j++) //loop to feed the data of the vertices to the cell
						{
						cells.back().vertices[j] = vertex_indices[cells.back().vertices[j]];
							}
						cell_indices[count] = count;
						no_cells++;
					}
					else if (type == 2)
					{
						//If this is encountered, the pointer comes out of the loop
						//and starts processing boundaries.
						subcelldata.boundary_lines.push_back(CellData<1>());
						for (unsigned int j = 0; j < type; j++) //loop to feed the data to the boundary
						{
							in >> subcelldata.boundary_lines.back().vertices[j];
						}
						subcelldata.boundary_lines.back().material_id = 0;
						for (unsigned int j = 0; j < type; j++)
						{
							subcelldata.boundary_lines.back().vertices[j] = vertex_indices[subcelldata.boundary_lines.back().vertices[j]];
						}
						line_indices[no_lines] = no_lines + 1;
						no_lines++;
					}
					else
						AssertThrow (false, ExcMessage ("While reading VTK file, unknown file type encountered"));
				}
			}
			else
				AssertThrow (false, ExcMessage ("Dimension should be either 2 or 3"));
		}
		else
			AssertThrow (false, ExcMessage ("While reading VTK file, failed to find CELLS section"));

		/////////////////////Processing the CELL_TYPES section////////////////////////

		in >> keyword;

		if (keyword == "CELL_TYPES")//Entering the cell_types section and ignoring data.
			{
			in.ignore(256, '\n');
			while (!in.eof())
			{
				in>>keyword;
				if (keyword != "12" && keyword != "9")
				{
					break;
				}
			}
		}
	}
	else { // STRUCTURED_GRID
		Assert(dim == 3, ExcNotImplemented());
		total_cells = (N[0]-1) * (N[2]-1) * (N[2]-1);
		cells.resize(total_cells);
		unsigned int ci = 0;
		for (unsigned int nz=0; nz<N[2]-1; ++nz) {
			for (unsigned int ny=0; ny<N[1]-1; ++ny) {
				for (unsigned int nx=0; nx<N[0]-1; ++nx) {
					cells[ci].vertices[0] = N[0]*N[1]*nz + ny*N[0] + nx;
					cells[ci].vertices[1] = cells[ci].vertices[0] + 1;
					cells[ci].vertices[2] = cells[ci].vertices[0] + N[0];
					cells[ci].vertices[3] = cells[ci].vertices[2] + 1;
					cells[ci].vertices[4] = cells[ci].vertices[0] + N[0]*N[1];
					cells[ci].vertices[5] = cells[ci].vertices[4] + 1;
					cells[ci].vertices[6] = cells[ci].vertices[4] + N[0];
					cells[ci].vertices[7] = cells[ci].vertices[6] + 1;
					++ci;
					cell_indices[ci] = ci;
				}
			}
		}
		no_cells = total_cells;
	}

	////////////////////////Processing the CELL_DATA section/////////////////////////////

	if (keyword == "CELL_DATA")
	{
		int no_ids;
		in>>no_ids;

		std::string linenew;
		std::string textnew[2];
		textnew[0] = "SCALARS PORO float";
		textnew[1] = "LOOKUP_TABLE default";

		in.ignore(256, '\n');

		for (unsigned int i = 0; i < 2; i++)
		{
			getline(in, linenew);
			if (i == 0)
				if (linenew.size() > textnew[0].size())
					linenew.resize(textnew[0].size());
			if (linenew.compare(textnew[i]) != 0) {
				std::cout << "Warning: While reading VTK file, failed to find <" << textnew[i] << "> section" << std::endl;
				break;
			}
		}

		poro.resize(no_cells);
		for (unsigned int i = 0; i < no_cells; i++) //assigning IDs to cells.
		{
			double id;
			in>>id;
			poro[cell_indices[i]] = id;
		}
		have_poro = true;

		in.ignore(256, '\n');
		std::string keystring = "FIELD";
		getline(in, linenew);
		if (linenew.size() > keystring.size())
			linenew.resize(keystring.size());
		if (linenew.compare(keystring) != 0)
			std::cout << "Warning: While reading VTK file, failed to find <" << keystring << "> section" << std::endl
					  << "Unable to read PERM*" << std::endl;

		char dir = 'X';
		for (int d=0; d<dim; ++d, ++dir) {
			keystring = "PERM";
			keystring.append(1, dir);
			std::string stringnew;
			in >> stringnew;
			if (stringnew.compare(keystring) != 0) {
				std::cout << "Warning: While reading VTK file, failed to find <" << keystring << "> section" << std::endl;
				break;
			}
			in >> stringnew; // ignore next
			double n_vals;
			in >> n_vals;
			if ( no_cells != n_vals ) {
				std::cout << "In VTK file: Nr of entries for " << keystring << " is not equal to nr of cells." << std::endl;
				exit(1);
			}
			in >> stringnew; // ignore next
			perm.resize(d+1);
			perm[d].resize(no_cells);
			for (unsigned int i = 0; i < no_cells; i++) //assigning IDs to cells.
			{
				double id;
				in>>id;
				perm[d][cell_indices[i]] = id;
			}
			have_perm[d] = true;
			in.ignore(256, '\n');
		}
	}
	else
		std::cout << "Keyword CELL_DATA not found\n";

	Assert(subcelldata.check_consistency(dim), ExcInternalError());

	GridTools::delete_unused_vertices(vertices,
									  cells,
									  subcelldata);

	/*
	GridReordering<dim, dim>::invert_all_cells_of_negative_grid(vertices, cells);
	GridReordering<dim, dim>::reorder_cells(cells);
	triangulation->create_triangulation_compatibility(vertices,
													  cells,
													  subcelldata);
	*/
	triangulation->create_triangulation(vertices,
										cells,
										subcelldata);
}


#endif // VTK_READER_H
