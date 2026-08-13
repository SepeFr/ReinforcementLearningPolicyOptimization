#include "LatinHypercubeSampler.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

Eigen::MatrixXd LatinHypercubeSampler::generate( std::size_t point_count, std::size_t dimension )
{
  if ( point_count == 0 || dimension == 0 )
  {
    throw std::invalid_argument( "LatinHypercubeSampler: point count and dimension must be greater than zero" );
  }

  Eigen::MatrixXd design( static_cast< Eigen::Index >( point_count ), static_cast< Eigen::Index >( dimension ) );
  std::uniform_real_distribution< double > within_stratum( 0.0, 1.0 );
  std::vector< std::size_t > strata( point_count );

  for ( std::size_t coordinate = 0; coordinate < dimension; ++coordinate )
  {
    std::iota( strata.begin(), strata.end(), std::size_t{ 0 } );
    std::shuffle( strata.begin(), strata.end(), generator_ );

    for ( std::size_t point = 0; point < point_count; ++point )
    {
      // u_ij = (pi_j(i) + xi_ij) / point_count, with xi_ij sampled in [0, 1).
      design( static_cast< Eigen::Index >( point ), static_cast< Eigen::Index >( coordinate ) ) =
        ( static_cast< double >( strata.at( point ) ) + within_stratum( generator_ ) ) /
        static_cast< double >( point_count );
    }
  }

  return design;
}
