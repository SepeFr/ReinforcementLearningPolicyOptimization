#include "NeighborhoodStrategy.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{
  // Acklam's rational approximation of Phi^{-1}(p). LHS values are clamped
  // away from 0 and 1 because the normal quantile is infinite at both ends.
  double inverseStandardNormalCDF( double probability )
  {
    constexpr double a1 = -3.969683028665376e+01;
    constexpr double a2 = 2.209460984245205e+02;
    constexpr double a3 = -2.759285104469687e+02;
    constexpr double a4 = 1.383577518672690e+02;
    constexpr double a5 = -3.066479806614716e+01;
    constexpr double a6 = 2.506628277459239e+00;
    constexpr double b1 = -5.447609879822406e+01;
    constexpr double b2 = 1.615858368580409e+02;
    constexpr double b3 = -1.556989798598866e+02;
    constexpr double b4 = 6.680131188771972e+01;
    constexpr double b5 = -1.328068155288572e+01;
    constexpr double c1 = -7.784894002430293e-03;
    constexpr double c2 = -3.223964580411365e-01;
    constexpr double c3 = -2.400758277161838e+00;
    constexpr double c4 = -2.549732539343734e+00;
    constexpr double c5 = 4.374664141464968e+00;
    constexpr double c6 = 2.938163982698783e+00;
    constexpr double d1 = 7.784695709041462e-03;
    constexpr double d2 = 3.224671290700398e-01;
    constexpr double d3 = 2.445134137142996e+00;
    constexpr double d4 = 3.754408661907416e+00;
    constexpr double lower_region = 0.02425;
    constexpr double upper_region = 1.0 - lower_region;

    const double epsilon = std::numeric_limits< double >::epsilon();
    const double p = std::clamp( probability, epsilon, 1.0 - epsilon );

    if ( p < lower_region )
    {
      const double q = std::sqrt( -2.0 * std::log( p ) );
      return ( ( ( ( ( c1 * q + c2 ) * q + c3 ) * q + c4 ) * q + c5 ) * q + c6 ) /
        ( ( ( ( d1 * q + d2 ) * q + d3 ) * q + d4 ) * q + 1.0 );
    }
    if ( p <= upper_region )
    {
      const double q = p - 0.5;
      const double r = q * q;
      return ( ( ( ( ( a1 * r + a2 ) * r + a3 ) * r + a4 ) * r + a5 ) * r + a6 ) * q /
        ( ( ( ( ( b1 * r + b2 ) * r + b3 ) * r + b4 ) * r + b5 ) * r + 1.0 );
    }

    const double q = std::sqrt( -2.0 * std::log( 1.0 - p ) );
    return -( ( ( ( ( c1 * q + c2 ) * q + c3 ) * q + c4 ) * q + c5 ) * q + c6 ) /
      ( ( ( ( d1 * q + d2 ) * q + d3 ) * q + d4 ) * q + 1.0 );
  }
} // namespace

LatinHypercubeUniformNeighborhood::LatinHypercubeUniformNeighborhood(
  LatinHypercubeUniformNeighborhoodConfiguration configuration, std::size_t seed ) :
    configuration_( std::move( configuration ) ), sampler_( seed )
{
  if ( configuration_.lower_bound > configuration_.upper_bound )
  {
    throw std::invalid_argument( "LatinHypercubeUniformNeighborhood: lower bound cannot exceed upper bound" );
  }
}


Eigen::VectorXd
LatinHypercubeUniformNeighborhood::generateNeighbor( Eigen::Ref< const Eigen::VectorXd > current_parameters )
{
  return generateNeighbors( current_parameters, 1 ).front();
}


std::vector< Eigen::VectorXd >
LatinHypercubeUniformNeighborhood::generateNeighbors( Eigen::Ref< const Eigen::VectorXd > current_parameters,
                                                      std::size_t number_of_neighbors )
{
  if ( number_of_neighbors == 0 )
  {
    return {};
  }

  const Eigen::MatrixXd design =
    sampler_.generate( number_of_neighbors, static_cast< std::size_t >( current_parameters.size() ) );
  std::vector< Eigen::VectorXd > neighbors;
  neighbors.reserve( number_of_neighbors );

  for ( Eigen::Index row = 0; row < design.rows(); ++row )
  {
    const Eigen::VectorXd displacement =
      ( configuration_.lower_bound +
        ( configuration_.upper_bound - configuration_.lower_bound ) * design.row( row ).array() )
        .matrix()
        .transpose();
    neighbors.push_back( current_parameters + displacement );
  }

  return neighbors;
}


void LatinHypercubeUniformNeighborhood::reset() { sampler_.reset(); }


LatinHypercubeGaussianNeighborhood::LatinHypercubeGaussianNeighborhood(
  LatinHypercubeGaussianNeighborhoodConfiguration configuration, std::size_t seed ) :
    configuration_( std::move( configuration ) ), sampler_( seed )
{
  if ( !std::isfinite( configuration_.standard_deviation ) || configuration_.standard_deviation <= 0.0 )
  {
    throw std::invalid_argument( "LatinHypercubeGaussianNeighborhood: standard deviation must be finite and positive" );
  }
}


Eigen::VectorXd
LatinHypercubeGaussianNeighborhood::generateNeighbor( Eigen::Ref< const Eigen::VectorXd > current_parameters )
{
  return generateNeighbors( current_parameters, 1 ).front();
}


std::vector< Eigen::VectorXd >
LatinHypercubeGaussianNeighborhood::generateNeighbors( Eigen::Ref< const Eigen::VectorXd > current_parameters,
                                                       std::size_t number_of_neighbors )
{
  if ( number_of_neighbors == 0 )
  {
    return {};
  }

  const Eigen::MatrixXd design =
    sampler_.generate( number_of_neighbors, static_cast< std::size_t >( current_parameters.size() ) );
  std::vector< Eigen::VectorXd > neighbors;
  neighbors.reserve( number_of_neighbors );

  for ( Eigen::Index row = 0; row < design.rows(); ++row )
  {
    Eigen::VectorXd displacement( current_parameters.size() );
    for ( Eigen::Index coordinate = 0; coordinate < displacement.size(); ++coordinate )
    {
      const double standard_normal = inverseStandardNormalCDF( design( row, coordinate ) );
      displacement( coordinate ) =
        configuration_.perturbation_mean + configuration_.standard_deviation * standard_normal;
    }
    neighbors.push_back( current_parameters + displacement );
  }

  return neighbors;
}


void LatinHypercubeGaussianNeighborhood::reset() { sampler_.reset(); }
