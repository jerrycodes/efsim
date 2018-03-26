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

#ifndef PROBLEM_FUNCTIONS_H
#define PROBLEM_FUNCTIONS_H

#include <deal.II/grid/tria.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/function_time.h>
#include <deal.II/base/parameter_handler.h>

#include "BitMapFunction.h"
#include <boost/concept_check.hpp>

#include <cfloat>


// This file contains problem spesific functions


const double PI = 4*atan(1);

// List of valid problems
enum ProblemType { VTK, SIMPLE_ANALYTIC, LOWPERM_REGION, ANALYTIC, ANALYTIC_STEADYP, BITMAP, ONED, CHANNEL, 
	               FRACTURE_ANALYTIC, REGULAR_NETWORK, REGULAR_NETWORK_RESOLVED, SIMPLE_FRAC, SIMPLE_FRAC_RESOLVED, COMPLEX_NETWORK };

// Get function to retrive ProblemType from ParameterHandler
ProblemType getProblemType(ParameterHandler& prm,
						   const std::string entry,
		                   const std::string subsection = "") {
	const bool subsection_given = ( ! subsection.empty() );
	if (subsection_given)
		prm.enter_subsection(subsection);
	std::string problem = prm.get(entry);
	ProblemType pt;
	if (problem == "LOWPERM_REGION")
		pt = ProblemType::LOWPERM_REGION;
	else if (problem == "SIMPLE_ANALYTIC")
		pt = ProblemType::SIMPLE_ANALYTIC;
	else if (problem == "ANALYTIC")
		pt = ProblemType::ANALYTIC;
	else if (problem == "ANALYTIC_STEADYP")
		pt = ProblemType::ANALYTIC_STEADYP;
	else if (problem == "BITMAP")
		pt = ProblemType::BITMAP;
	else if (problem == "ONED")
		pt = ProblemType::ONED;
	else if (problem == "CHANNEL")
		pt = ProblemType::CHANNEL;
	else if (problem == "FRACTURE_ANALYTIC")
		pt = ProblemType::FRACTURE_ANALYTIC;
	else if (problem == "REGULAR_NETWORK")
		pt = ProblemType::REGULAR_NETWORK;
	else if (problem == "REGULAR_NETWORK_RESOLVED")
		pt = ProblemType::REGULAR_NETWORK_RESOLVED;
	else if (problem == "SIMPLE_FRAC")
		pt = ProblemType::SIMPLE_FRAC;
	else if (problem == "COMPLEX_NETWORK")
		pt = ProblemType::COMPLEX_NETWORK;
	else if (problem == "SIMPLE_FRAC_RESOLVED")
		pt = ProblemType::SIMPLE_FRAC_RESOLVED;
	else {
		std::cout << "Warning: Field 'ProblemType' from input is not recognized (see ProblemFunctions.h for valid options).\n"
				  << "         Using default: LOWPERM_REGION.\n";
		pt = ProblemType::LOWPERM_REGION;
	}
	if (subsection_given)
		prm.leave_subsection();
	return pt;
}



/////////////////////////////////////
// ExactPressure
/////////////////////////////////////

template <int dim, ProblemType pt>
class ExactPressure : public Function<dim> 
{
public:
	ExactPressure() : Function<dim>() {}
	virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim, ProblemType pt>
double ExactPressure<dim,pt>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
	Assert(false, ExcNotImplemented());
	return 0.0;
}

template <>
double ExactPressure<2,ONED>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	return 1-p(0)*p(0);
}

template <>
double ExactPressure<2,SIMPLE_ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	return sin(PI*p(0)) * cos(2*PI*p(1)) * cos(this->get_time());
}

template <>
double ExactPressure<2,ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	return std::cos(this->get_time() + p(0) - p(1));
}

template <>
double ExactPressure<3,ANALYTIC>::value(const Point<3> &p, const unsigned int /*component*/) const
{
	return std::cos(this->get_time() + p(0) - p(1) + p(2));
}

template <>
double ExactPressure<2,ANALYTIC_STEADYP>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	return std::cos(p(0) - p(1));
}

template <>
double ExactPressure<3,ANALYTIC_STEADYP>::value(const Point<3> &p, const unsigned int /*component*/) const
{
	return std::cos(p(0) - p(1) + p(2));
}

template <>
double ExactPressure<2,FRACTURE_ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double e = exp(1);
	const double r = p.norm();
	Assert(r>=1, ExcMessage("Function undefined for r<1"));
	if (r <= e)
		return log(r) * (4.0+e) / 5.0;
	else
		return (4-4*e) * (log(r)-5.0/4.0) / 5.0 + 1.0;
}



/////////////////////////////////////
// LinearDecreasing
/////////////////////////////////////

template <int dim>
class LinearDecreasing : public Function<dim> 
{
public:
	LinearDecreasing(int dir, double magnitude_ = 1.0); 
	LinearDecreasing(int dir, const Triangulation<dim> &tria, double magnitude_ = 1.0);
	LinearDecreasing(int dir, double coord_min, double coord_max, double magnitude_ = 1.0);
	virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
private:
	int    direction;
	double magnitude;
	double range_min;
	double range_max;
};

template <int dim>
LinearDecreasing<dim>::LinearDecreasing(int dir, double magnitude_)
: Function<dim>(), direction(dir), magnitude(magnitude_), range_min(0.0), range_max(1.0)
{
	Assert(direction < dim, ExcInternalError());
}

template <int dim>
LinearDecreasing<dim>::LinearDecreasing(int dir, const Triangulation<dim> &tria, double magnitude_)
: LinearDecreasing<dim>(dir, magnitude_) 
{
	// Assuming shoe-box grid
	range_min =  DBL_MAX;
	range_max = -DBL_MAX;
	
	typename Triangulation<dim>::active_cell_iterator
	cell = tria.begin_active(),
	endc = tria.end();
	for (; cell!=endc; ++cell) {
		for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face) {
			if (cell->at_boundary(face)) {
				const double face_coord = cell->face(face)->center()(direction);
				if ( face_coord < range_min )
					range_min = face_coord;
				if ( face_coord > range_max )
					range_max = face_coord;
			}
		}
	}
}

template <int dim>
LinearDecreasing<dim>::LinearDecreasing(int dir, double coord_min, double coord_max, double magnitude_)
: Function<dim>(), direction(dir), magnitude(magnitude_), range_min(coord_min), range_max(coord_max)
{
	Assert(direction < dim, ExcInternalError());
}

template <int dim>
double LinearDecreasing<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
	return magnitude * (range_max - p(direction)) / (range_max - range_min);
}


/////////////////////////////////////
// ExactGradient
/////////////////////////////////////

template <int dim, ProblemType pt>
class ExactGradient : public TensorFunction<1,dim> 
{
public:
	ExactGradient() : TensorFunction<1,dim>() {}
	virtual Tensor<1,dim> value(const Point<dim> &p) const;
};

template <int dim, ProblemType pt>
Tensor<1,dim> ExactGradient<dim,pt>::value(const Point<dim> & /*p*/) const
{
	Assert(false, ExcNotImplemented());
	return Tensor<1,dim>();
}

template <>
Tensor<1,2> ExactGradient<2,ONED>::value(const Point<2> &p) const
{
	Tensor<1,2> return_value;
	return_value[0] = -2.0*p(0);
	return_value[1] = 0.0;
	return return_value;
}

template <>
Tensor<1,2> ExactGradient<2,SIMPLE_ANALYTIC>::value(const Point<2> &p) const
{
	Tensor<1,2> return_value;
	return_value[0] = PI * std::cos(PI*p(0)) * std::cos(2*PI*p(1));
	return_value[1] = -2*PI * std::sin(PI*p(0)) * std::sin(2*PI*p(1));
	return_value *= cos(this->get_time());
	return return_value;
}

template <>
Tensor<1,2> ExactGradient<2,ANALYTIC>::value(const Point<2> &p) const
{
	Tensor<1,2> return_value;
	const double comp = std::sin(this->get_time() + p(0) - p(1));
	return_value[0] = -comp;
	return_value[1] =  comp;
	return return_value;
}

template <>
Tensor<1,3> ExactGradient<3,ANALYTIC>::value(const Point<3> &p) const
{
	Tensor<1,3> return_value;
	const double comp = std::sin(this->get_time() + p(0) - p(1) + p(2));
	return_value[0] = -comp;
	return_value[1] =  comp;
	return_value[2] = -comp;
	return return_value;
}

template <>
Tensor<1,2> ExactGradient<2,ANALYTIC_STEADYP>::value(const Point<2> &p) const
{
	Tensor<1,2> return_value;
	const double comp = std::sin(p(0) - p(1));
	return_value[0] = -comp;
	return_value[1] =  comp;
	return return_value;
}

template <>
Tensor<1,3> ExactGradient<3,ANALYTIC_STEADYP>::value(const Point<3> &p) const
{
	Tensor<1,3> return_value;
	const double comp = std::sin(p(0) - p(1) + p(2));
	return_value[0] = -comp;
	return_value[1] =  comp;
	return_value[2] = -comp;
	return return_value;
}

template <>
Tensor<1,2> ExactGradient<2,FRACTURE_ANALYTIC>::value(const Point<2> &p) const
{
	const double e = exp(1);
	const double r = p.norm();
	Assert(r>=1, ExcMessage("Function undefined for r<1"));

	double factor;
	if (r <= e)
		factor = (4.0+e) / (5.0*r*r);
	else
		factor = (4.0-4.0*e) / (5.0*r*r);

	return factor * p;
}



template <int dim>
class ZeroGradientFunction : public ZeroTensorFunction<1,dim>
{};


/////////////////////////////////////
// RightHandSide
/////////////////////////////////////

template <int dim, ProblemType pt>
class RightHandSide : public Function<dim>
{
public:
	RightHandSide() : Function<dim>() {}
	virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim, ProblemType pt>
double RightHandSide<dim,pt>::value(const Point<dim>& /*p*/, const unsigned int /*component*/) const
{
	return 0.0;
}

template <>
double RightHandSide<2,ONED>::value(const Point<2>& /*p*/, const unsigned int /*component*/) const
{
	return 2.0;
}

template <>
double RightHandSide<2,SIMPLE_ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	// 5*PI^2 - sin(t) * pressure(x,y,t)
	return (5*PI*PI - sin(this->get_time())) * (sin(PI*p(0)) * cos(2*PI*p(1)) * cos(this->get_time()));
}

template <>
double RightHandSide<2,ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	return 2*cos(this->get_time() + p(0) - p(1)) - sin(this->get_time() + p(0) - p(1));
}

template <>
double RightHandSide<3,ANALYTIC>::value(const Point<3> &p, const unsigned int /*component*/) const
{
	return 3*cos(this->get_time() + p(0) - p(1) + p(2)) - sin(this->get_time() + p(0) - p(1) + p(2));
}

template <>
double RightHandSide<2,ANALYTIC_STEADYP>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	return 2*cos(p(0) - p(1));
}

template <>
double RightHandSide<3,ANALYTIC_STEADYP>::value(const Point<3> &p, const unsigned int /*component*/) const
{
	return 3*cos(p(0) - p(1) + p(2));
}


/////////////////////////////////////
// WellFunction
/////////////////////////////////////

template <int dim>
class WellFunction : public Function<dim>
{
public:
	WellFunction(double w, double r)
	: Function<dim>(),
	  width(w),
	  rate(r)
	{}
	virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
private:
	double width;
	double rate;
};

template <int dim>
double WellFunction<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
	if (p(0) <= width && p(1) <= width)
		return rate;
	else if (p(0) >= 1-width && p(1) >= 1-width)
		return -rate;
	return 0.0;
}


/////////////////////////////////////
// PermeabilityTensor
/////////////////////////////////////

template<int dim, ProblemType pt>
class PermeabilityTensor : public TensorFunction<2,dim>
{
public:
	PermeabilityTensor();
	PermeabilityTensor(std::string pgmfile, double perm_min = 1e-2, double perm_max = 1e2);
	virtual Tensor<2,dim> value(const Point<dim> &p) const;
private:
	Function<dim>* logkfun;
};

template <int dim, ProblemType pt>
PermeabilityTensor<dim,pt>::PermeabilityTensor()
: TensorFunction<2,dim> ()
{}

template <int dim, ProblemType pt>
PermeabilityTensor<dim,pt>::PermeabilityTensor(std::string pgmfile, double perm_min, double perm_max)
: TensorFunction<2,dim> ()
{
	const double perm_min_log = log10(perm_min);
	const double perm_max_log = log10(perm_max);
	logkfun = new BitmapFunction<dim>(pgmfile, 0.0, 1.0, 0.0, 1.0, perm_min_log, perm_max_log);
}

template <int dim, ProblemType pt>
Tensor<2,dim> PermeabilityTensor<dim,pt>::value(const Point<dim> & /*p*/) const
{
	Tensor<2,dim> identity;
	for (int i=0; i<dim; ++i)
		identity[i][i] = 1.0;
	return Tensor<2,dim>();
}

template<>
Tensor<2,2> PermeabilityTensor<2,LOWPERM_REGION>::value(const Point<2> &p) const
{
	Tensor<2,2> return_tensor;
	double k = 1.0;
	if (  ( (p(0)>3.0/8.0) && (p(0)<5.0/8.0) ) && ( (p(1)>1.0/4.0) && (p(1)<3.0/4.0) ) )
		k = 1e-3;
	for (int i=0; i<2; ++i)
		return_tensor[i][i] = k;
	return return_tensor;
}

template<>
Tensor<2,3> PermeabilityTensor<3,LOWPERM_REGION>::value(const Point<3> &p) const
{
	Tensor<2,3> return_tensor;
	double k = 1.0;
	if (  ( (p(0)>3.0/8.0) && (p(0)<5.0/8.0) ) && ( (p(1)>1.0/4.0) && (p(1)<3.0/4.0) ) )
		k = 1e-3;
	if ( p(2) > 3.0/4.0 ) 
		k = 1.0;
	for (int i=0; i<3; ++i)
		return_tensor[i][i] = k;
	return return_tensor;
}

template<>
Tensor<2,2> PermeabilityTensor<2,BITMAP>::value(const Point<2> &p) const
{
	Tensor<2,2> return_tensor;
	double k = pow(10, logkfun->value(p));
	for (int i=0; i<2; ++i)
		return_tensor[i][i] = k;
	return return_tensor;
}

template<>
Tensor<2,2> PermeabilityTensor<2,CHANNEL>::value(const Point<2> &p) const
{
	Tensor<2,2> return_tensor;
	double k = pow(10, logkfun->value(p));
	for (int i=0; i<2; ++i)
		return_tensor[i][i] = k;
	return return_tensor;
}

template<>
Tensor<2,2> PermeabilityTensor<2,REGULAR_NETWORK_RESOLVED>::value(const Point<2> &p) const
{
	const double w = 1e-4;
	const double k_frac = 1e4;
	
	Tensor<2,2> return_tensor;
	double k = 1.0;
	const double x = p(0);
	const double y = p(1);
	
	if ( y > 0.5-w/2.0 && y <  0.5+w/2.0 )
		k = k_frac;
	else if ( x > 0.5-w/2.0 && x <  0.5+w/2.0 )
		k = k_frac;
	else if ( x > 0.5 && (y > 0.75-w/2.0 && y <  0.75+w/2.0) )
		k = k_frac;
	else if ( y > 0.5 && (x > 0.75-w/2.0 && x <  0.75+w/2.0) )
		k = k_frac;
	else if ( (x > 0.5 && x < 0.75) && (y > 0.625-w/2.0 && y <  0.625+w/2.0) )
		k = k_frac;
	else if ( (y > 0.5 && y < 0.75) && (x > 0.625-w/2.0 && x <  0.625+w/2.0) )
		k = k_frac;
		
	for (int i=0; i<2; ++i)
		return_tensor[i][i] = k;
	return return_tensor;
}


template<int dim>
class IdentityTensorFunction : public TensorFunction<2,dim>
{
public:
	IdentityTensorFunction() : TensorFunction<2,dim>() {} 
	virtual Tensor<2,dim> value(const Point<dim> &p) const;
};

template<int dim>
Tensor<2,dim> IdentityTensorFunction<dim>::value(const Point<dim> & /*p*/) const
{
	Tensor<2,dim> return_tensor;
	double k = 1.0;
	for (int i=0; i<dim; ++i)
		return_tensor[i][i] = k;
	return return_tensor;
}


/////////////////////////////////////
// Neumann
/////////////////////////////////////

template <int dim, ProblemType pt>
class Neumann : public Function<dim>
{
public:
	Neumann() : Function<dim>() {}
	virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim, ProblemType pt>
double Neumann<dim,pt>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
	return 0.0;
}

template <>
double Neumann<2,ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double x = p(0);
	const double y = p(1);
	const double t = this->get_time();
	if ( y == 1.0 ) // Top
		return -sin(t+x-1);
	else if (y == 0.0) // Bottom
		return sin(t+x);
	else if (x == 0.0)  // Left
		return -sin(t-y);
	else if (x == 1.0) // Right
		return sin(t-y+1);
	Assert(false, ExcMessage("Should not need Neumann conditions away from boundary."));
	return 0.0;
}

template <>
double Neumann<3,ANALYTIC>::value(const Point<3> &p, const unsigned int /*component*/) const
{
	const double x = p(0);
	const double y = p(1);
	const double z = p(2);
	const double t = this->get_time();
	if (z == 1.0) // Top
		return sin(t+x-y+1);
	else if (z == 0.0) // Bottom
		return -sin(t+x-y);
	else if (y == 1.0) // Back
		return -sin(t+x+z-1);
	else if (y == 0.0) // Front
		return sin(t+x+z);
	else if (x == 0.0)  // Left
		return -sin(t-y+z);
	else if (x == 1.0) // Right
		return sin(t-y+z+1);
	Assert(false, ExcMessage("Should not need Neumann conditions away from boundary."));
	return 0.0;
}

template <>
double Neumann<2,ANALYTIC_STEADYP>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double x = p(0);
	const double y = p(1);
	if ( y == 1.0 ) // Top
	return -sin(x-1);
	else if (y == 0.0) // Bottom
		return sin(x);
	else if (x == 0.0)  // Left
		return -sin(-y);
	else if (x == 1.0) // Right
		return sin(-y+1);
	Assert(false, ExcMessage("Should not need Neumann conditions away from boundary."));
	return 0.0;
}

template <>
double Neumann<3,ANALYTIC_STEADYP>::value(const Point<3> &p, const unsigned int /*component*/) const
{
	const double x = p(0);
	const double y = p(1);
	const double z = p(2);
	if (z == 1.0) // Top
		return sin(x-y+1);
	else if (z == 0.0) // Bottom
		return -sin(x-y);
	else if (y == 1.0) // Back
		return -sin(x+z-1);
	else if (y == 0.0) // Front
		return sin(x+z);
	else if (x == 0.0)  // Left
		return -sin(-y+z);
	else if (x == 1.0) // Right
		return sin(-y+z+1);
	Assert(false, ExcMessage("Should not need Neumann conditions away from boundary."));
	return 0.0;
}

template <>
double Neumann<2,ONED>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double x = p(0);
	if (x == 1.0) // Right
		return 2.0;
	return 0.0;
}

template <>
double Neumann<2,SIMPLE_ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double x = p(0);
	if (x == 1.0 || x == 0.0) // Right
		return PI * cos(2*PI*p(1));
	return 0.0;
}

template <>
double Neumann<2,REGULAR_NETWORK>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double x = p(0);
	if (x == 0.0) // Left
		return -1.0;
	return 0.0;
}



/////////////////////////////////////
// ExactConcentration
/////////////////////////////////////

template <int dim, ProblemType pt>
class ExactConcentration : public Function<dim>
{
public:
	ExactConcentration() : Function<dim>() {}
	virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim, ProblemType pt>
double ExactConcentration<dim,pt>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
	Assert(false, ExcNotImplemented());
	return 0.0;
}

template <>
double ExactConcentration<2,ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	return std::sin(t + p(0) - p(1));
}

template <>
double ExactConcentration<3,ANALYTIC>::value(const Point<3> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	return std::sin(t + p(0) - p(1) + p(2));
}

template <>
double ExactConcentration<2,SIMPLE_ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	return sin(PI*p(0)) * exp(t-p(1));
}

template <>
double ExactConcentration<2,ONED>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	return sin(t + p(0) - p(1));
}



/////////////////////////////////////
// RightHandSideTransport
/////////////////////////////////////

template <int dim, ProblemType pt>
class RightHandSideTransport : public Function<dim>
{
public:
	RightHandSideTransport() : Function<dim>() {}
	virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim, ProblemType pt>
double RightHandSideTransport<dim,pt>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
	return 0.0;
}

template <>
double RightHandSideTransport<2,ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	const double x = p(0);
	const double y = p(1);
	return cos(t+x-y) + 4*sin(t+x-y)*cos(t+x-y);
}

template <>
double RightHandSideTransport<3,ANALYTIC>::value(const Point<3> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	const double x = p(0);
	const double y = p(1);
	const double z = p(2);
	return cos(t+x-y+z) + 6*sin(t+x-y+z)*cos(t+x-y+z);
}

template <>
double RightHandSideTransport<2,SIMPLE_ANALYTIC>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	const double x = p(0);
	const double y = p(1);
	return exp(t-y)*( sin(PI*x) - PI*PI*cos(t)*cos(PI*x)*cos(PI*x)*cos(2*PI*y)
					  - 2*PI*cos(t)*sin(PI*x)*sin(PI*x)*sin(2*PI*y)
					  + 5*PI*PI*sin(PI*x)*sin(PI*x)*cos(2*PI*y)*cos(t)
					  - sin(t)*sin(PI*x)*cos(2*PI*y)*cos(t)*sin(PI*x)
					  + sin(PI*x)*sin(PI*x)*cos(2*PI*y)*sin(t) );
}


/////////////////////////////////////
// BoundaryConcentration
/////////////////////////////////////

template <int dim, ProblemType pt>
class BoundaryConcentration : public Function<dim>
{
public:
	BoundaryConcentration(double inflow_c = 1.0) : Function<dim>(), inflow_conc(inflow_c) {}
	virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
private:
	double inflow_conc;
};

template <int dim, ProblemType pt>
double BoundaryConcentration<dim,pt>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
	return inflow_conc;
}

template <>
double BoundaryConcentration<2,CHANNEL>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	double val = 0.0;
	if (p(0) == 0.0 && (p(1) < 0.25 && p(1) > 0.125) )
		val = 1.0;
	return inflow_conc * val;
}


/////////////////////////////////////
// WellConcentration
/////////////////////////////////////

template <int dim, ProblemType pt>
class WellConcentration : public Function<dim>
{
public:
	WellConcentration() : Function<dim>() {}
	virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim, ProblemType pt>
double WellConcentration<dim,pt>::value(const Point<dim> &/*p*/, const unsigned int /*component*/) const
{
	return 0.0;
}

template <>
double WellConcentration<2,ANALYTIC_STEADYP>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	if ( t > 0.105)
		Assert(false, ExcMessage("The example case ANALYTIC_STEADYP does only work for t ~< 0.1."));
	const double x = p(0);
	const double y = p(1);
	return cos(t+x-y)/(2*cos(x-y)) + sin(t+x-y) + tan(x-y)*cos(t+x-y);
}

template <>
double WellConcentration<3,ANALYTIC_STEADYP>::value(const Point<3> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	if ( t > 0.105)
		Assert(false, ExcMessage("The example case ANALYTIC_STEADYP does only work for t ~< 0.1."));
	const double x = p(0);
	const double y = p(1);
	const double z = p(2);
	return cos(t+x-y+z)/(3*cos(x-y+z)) + sin(t+x-y+z) + tan(x-y+z)*cos(t+x-y+z);
}

template <>
double WellConcentration<2,ONED>::value(const Point<2> &p, const unsigned int /*component*/) const
{
	const double t = this->get_time();
	const double x = p(0);
	const double y = p(1);
	const double a = t+x-y;
	return (0.5+x)*cos(a) + sin(a);
}


/////////////////////////////////////
// Divergence of velocity
/////////////////////////////////////

template <int dim, ProblemType pt>
class DivVelocity : public Function<dim>
{
public:
	DivVelocity() : Function<dim>() {}
	virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim, ProblemType pt>
double DivVelocity<dim,pt>::value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
{
	Assert(false, ExcMessage("DivVelocity not implemented for all problem types."));
	return 0.0;
}

template <>
double DivVelocity<2,ANALYTIC_STEADYP>::value(const Point<2> & p, const unsigned int /*component*/) const
{
	return 2*cos(p(0)-p(1));
}

template <>
double DivVelocity<2,ONED>::value(const Point<2> & /*p*/, const unsigned int /*component*/) const
{
	return 2;
}


/////////////////////////////////////
// Fracture parametrization
/////////////////////////////////////

class FractureParametrization : Function<1>
{
public:
	FractureParametrization() : Function<1>(2) {}
	virtual double value(const Point<1> &p, const unsigned int component) const =0;
	double start() { return s0; }
	double end()   { return s1; }
	
protected:
	double s0, s1;
};


class FractureAnalytic : public FractureParametrization
{
public:
	FractureAnalytic() : FractureParametrization()
	{
		this->s0 = asin(exp(-1));
		this->s1 = acos(exp(-1));
	}
	
	virtual double value(const Point<1> &p, const unsigned int component) const;
private:
};

double FractureAnalytic::value(const Point<1> &p, const unsigned int component) const
{
	if (component == 0)
		return exp(1)*cos(p[0]);
	else
		return exp(1)*sin(p[0]);
}


class FractureArc : public FractureParametrization
{
public:
	FractureArc(Point<2> c, double r, double start, double stop)
	: FractureParametrization(), center(c), radius(r)
	{
		this->s0 = start;
		this->s1 = stop;
	}
	
	virtual double value(const Point<1> &p, const unsigned int component) const;
private:
	Point<2> center;
	double radius;
};

double FractureArc::value(const Point<1> &p, const unsigned int component) const
{
	if (component == 0) {
		return center[0] + radius * cos(p[0]);
	}
	else
		return center[1] + radius * sin(p[0]);
}


/////////////////////////////////////
// ProblemFunctionsFlow
/////////////////////////////////////

template <int dim>
struct ProblemFunctionsFlow
{
	ProblemFunctionsFlow() : analytic(false) {}
	ProblemFunctionsFlow(ProblemType pt);
	
	void set_problem(ProblemType pt);
	void set_linear_pressure(int dir, const Triangulation<dim> &tria, double magnitude = 1.0);
	
	void set_time(double time);
	
	bool analytic;
	
	Function<dim>*         exact_pressure   = new ZeroFunction<dim>();
	Function<dim>*         initial_pressure = new LinearDecreasing<dim>(0);
	TensorFunction<1,dim>* exact_gradient   = new ZeroGradientFunction<dim>();
	Function<dim>*         right_hand_side  = new ZeroFunction<dim>();
	Function<dim>*         dirichlet        = new LinearDecreasing<dim>(0);
	Function<dim>*         neumann          = new ZeroFunction<dim>();
	Function<dim>*         right_hand_side_fracture = new ZeroFunction<dim>();
};

template <int dim>
ProblemFunctionsFlow<dim>::ProblemFunctionsFlow(ProblemType pt)
{
	set_problem(pt);
}

template <int dim>
void ProblemFunctionsFlow<dim>::set_problem(ProblemType pt)
{
	switch(pt)
	{
		case ONED:
			Assert(dim == 2, ExcNotImplemented());
			analytic = true;
			exact_pressure   = new ExactPressure<dim,ONED>();
			initial_pressure = new ExactPressure<dim,ONED>();
			exact_gradient   = new ExactGradient<dim,ONED>();
			right_hand_side  = new RightHandSide<dim,ONED>();
			dirichlet        = exact_pressure;
			neumann          = new Neumann<dim,ONED>();
			break;
		case SIMPLE_ANALYTIC:
			Assert(dim == 2, ExcNotImplemented());
			analytic = true;
			exact_pressure   = new ExactPressure<dim,SIMPLE_ANALYTIC>();
			initial_pressure = new ExactPressure<dim,SIMPLE_ANALYTIC>();
			exact_gradient   = new ExactGradient<dim,SIMPLE_ANALYTIC>();
			right_hand_side  = new RightHandSide<dim,SIMPLE_ANALYTIC>();
			dirichlet        = exact_pressure;
			neumann          = new Neumann<dim,ONED>();
			break;
		case ANALYTIC:
			analytic = true;
			exact_pressure   = new ExactPressure<dim,ANALYTIC>();
			initial_pressure = new ExactPressure<dim,ANALYTIC>();
			exact_gradient   = new ExactGradient<dim,ANALYTIC>();
			right_hand_side  = new RightHandSide<dim,ANALYTIC>();
			dirichlet        = exact_pressure;
			neumann          = new Neumann<dim,ANALYTIC>();
			break;
		case ANALYTIC_STEADYP:
			analytic = true;
			exact_pressure   = new ExactPressure<dim,ANALYTIC_STEADYP>();
			exact_gradient   = new ExactGradient<dim,ANALYTIC_STEADYP>();
			right_hand_side  = new RightHandSide<dim,ANALYTIC_STEADYP>();
			dirichlet        = exact_pressure;
			neumann          = new Neumann<dim,ANALYTIC_STEADYP>();
			break;
		case FRACTURE_ANALYTIC:
			Assert(dim == 2, ExcNotImplemented());
			analytic = true;
			exact_pressure   = new ExactPressure<dim,FRACTURE_ANALYTIC>();
			exact_gradient   = new ExactGradient<dim,FRACTURE_ANALYTIC>();
			dirichlet        = exact_pressure;
			right_hand_side_fracture = new ConstantFunction<dim>(1.0);
			break;
		case REGULAR_NETWORK:
		case REGULAR_NETWORK_RESOLVED:
			Assert(dim == 2, ExcNotImplemented());
			analytic = false;
			dirichlet        = new ConstantFunction<dim>(1.0);
			neumann          = new Neumann<dim,REGULAR_NETWORK>();
			break;
		case CHANNEL:
		case BITMAP:
			Assert(dim == 2, ExcNotImplemented());
			analytic = false;
			break;
		default:
			analytic = false;
	}
}

template <int dim>
void ProblemFunctionsFlow<dim>::set_linear_pressure(int dir, const Triangulation<dim> &tria, double magnitude)
{
	analytic = false;
	initial_pressure = new LinearDecreasing<dim>(dir, tria, magnitude);
	dirichlet        = new LinearDecreasing<dim>(dir, tria, magnitude);
}

template <int dim>
void ProblemFunctionsFlow<dim>::set_time(double time)
{
	exact_pressure->set_time(time);
	exact_gradient->set_time(time);
	right_hand_side->set_time(time);
	dirichlet->set_time(time);
	neumann->set_time(time);
}


/////////////////////////////////////
// ProblemFunctionsRock
/////////////////////////////////////

template <int dim>
struct ProblemFunctionsRock
{
	ProblemFunctionsRock() {}
	ProblemFunctionsRock(ProblemType pt);
	
	void set_problem(ProblemType pt);
	
	Function<dim>*         porosity         = new ConstantFunction<dim>(1.0);
	TensorFunction<2,dim>* permeability     = new IdentityTensorFunction<dim>();
};

template <int dim>
ProblemFunctionsRock<dim>::ProblemFunctionsRock(ProblemType pt)
{
	set_problem(pt);
}

template <int dim>
void ProblemFunctionsRock<dim>::set_problem(ProblemType pt)
{
	switch(pt)
	{
		case LOWPERM_REGION:
			permeability = new PermeabilityTensor<dim,LOWPERM_REGION>();
			break;
		case BITMAP:
			permeability = new PermeabilityTensor<dim,BITMAP>("test.pgm", 1e-2, 1e2);
			break;
		case CHANNEL:
			permeability = new PermeabilityTensor<dim,CHANNEL>("channel.pgm", 1e-3, 1.0);
			break;
		case REGULAR_NETWORK_RESOLVED:
			permeability = new PermeabilityTensor<dim,REGULAR_NETWORK_RESOLVED>();
			break;
		default:
			break;
	}
}

/////////////////////////////////////
// ProblemFunctionsTransport
/////////////////////////////////////

template <int dim>
struct ProblemFunctionsTransport
{
	ProblemFunctionsTransport() {}
	ProblemFunctionsTransport(ProblemType pt);
	
	void set_problem(ProblemType pt);
	
	void set_time(double time);
	void set_inflow_concentration(double inflow);
	
	bool analytic;
	ProblemType problem;
	
	Function<dim>* exact_concentration       = new ZeroFunction<dim>();
	Function<dim>* initial_concentration     = new ZeroFunction<dim>();
	Function<dim>* right_hand_side           = new ZeroFunction<dim>();
	Function<dim>* right_hand_side_transport = new ZeroFunction<dim>();
	Function<dim>* boundary_concentration    = new ConstantFunction<dim>(1.0);
	Function<dim>* well_concentration        = new ConstantFunction<dim>(1.0);
	Function<dim>* udotn                     = new ZeroFunction<dim>();
	Function<dim>* div_velocity              = new ZeroFunction<dim>();
};

template <int dim>
ProblemFunctionsTransport<dim>::ProblemFunctionsTransport(ProblemType pt)
{
	set_problem(pt);
}

template <int dim>
void ProblemFunctionsTransport<dim>::set_problem(ProblemType pt)
{
	problem = pt;
	switch(pt)
	{
		case ONED:
			Assert(dim == 2, ExcNotImplemented());
			analytic = true;
			exact_concentration   = new ExactConcentration<dim,ONED>();
			initial_concentration = new ExactConcentration<dim,ONED>();
			well_concentration    = new WellConcentration<dim,ONED>();
			right_hand_side       = new RightHandSide<dim,ONED>();
			udotn                 = new Neumann<dim,ONED>();
			div_velocity          = new DivVelocity<dim,ONED>();
			break;
		case SIMPLE_ANALYTIC:
			Assert(dim == 2, ExcNotImplemented());
			analytic = true;
			exact_concentration       = new ExactConcentration<dim,SIMPLE_ANALYTIC>();
			initial_concentration     = new ExactConcentration<dim,SIMPLE_ANALYTIC>();
			right_hand_side_transport = new RightHandSideTransport<dim,SIMPLE_ANALYTIC>();
			boundary_concentration    = exact_concentration;
			udotn                     = new Neumann<dim,SIMPLE_ANALYTIC>();
			break;
		case ANALYTIC:
			analytic = true;
			exact_concentration       = new ExactConcentration<dim,ANALYTIC>();
			initial_concentration     = new ExactConcentration<dim,ANALYTIC>();
			right_hand_side_transport = new RightHandSideTransport<dim,ANALYTIC>();
			boundary_concentration    = exact_concentration;
			udotn                     = new Neumann<dim,ANALYTIC>();
			break;
		case ANALYTIC_STEADYP:
			analytic = true;
			exact_concentration    = new ExactConcentration<dim,ANALYTIC>();
			initial_concentration  = new ExactConcentration<dim,ANALYTIC>();
			right_hand_side        = new RightHandSide<dim,ANALYTIC_STEADYP>();
			boundary_concentration = exact_concentration;
			well_concentration     = new WellConcentration<dim,ANALYTIC_STEADYP>();
			div_velocity           = new DivVelocity<dim,ANALYTIC_STEADYP>();
			udotn                  = new Neumann<dim,ANALYTIC_STEADYP>();
			break;
		case CHANNEL:
			Assert(dim == 2, ExcNotImplemented());
			analytic = false;
			boundary_concentration = new BoundaryConcentration<dim,CHANNEL>();
			break;
		case SIMPLE_FRAC_RESOLVED:
			analytic = false;
			boundary_concentration = new LinearDecreasing<dim>(0);
			break;
		case COMPLEX_NETWORK:
			analytic = false;
			boundary_concentration = new LinearDecreasing<dim>(0, 0.0, 700.0);
			break;
		default:
			analytic = false;
	}
}

template <int dim>
void ProblemFunctionsTransport<dim>::set_time(double time)
{
	exact_concentration->set_time(time);
	right_hand_side->set_time(time);
	right_hand_side_transport->set_time(time);
	boundary_concentration->set_time(time);
	well_concentration->set_time(time);
}

template <int dim>
void ProblemFunctionsTransport<dim>::set_inflow_concentration(double inflow)
{
	if (problem == CHANNEL)
		boundary_concentration = new BoundaryConcentration<dim,CHANNEL>(inflow);
	else if (problem == SIMPLE_FRAC_RESOLVED)
		boundary_concentration = new LinearDecreasing<dim>(0, inflow);
	else if (problem == COMPLEX_NETWORK)
		boundary_concentration = new LinearDecreasing<dim>(0,0.0, 700.0, inflow);
	else if (!analytic)
		boundary_concentration = new ConstantFunction<dim>(inflow);
}



// Return boundary ids for faces in 2D rectangular domain
// 0 = Dirchlet
// 1 = Neumann
std::vector<unsigned int> get_boundary_id(ProblemType pt)
{
	std::vector<unsigned int> boundary_ids(4);
	
	boundary_ids[2] = 1;
	boundary_ids[3] = 1;
	
	if (pt == REGULAR_NETWORK || pt == REGULAR_NETWORK_RESOLVED)
		boundary_ids[0] = 1;
	
	return boundary_ids;
}


#endif // PROBLEM_FUNCTIONS_H