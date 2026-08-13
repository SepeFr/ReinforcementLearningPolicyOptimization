#include "FeedForwardNetworkConfiguration.hpp"
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{
  std::size_t checkedAdd( std::size_t left, std::size_t right )
  {
    if ( right > std::numeric_limits< std::size_t >::max() - left )
    {
      throw std::overflow_error( "FeedForwardNetworkConfiguration: parameter count addition overflow" );
    }
    return left + right;
  }

  std::size_t checkedMultiply( std::size_t left, std::size_t right )
  {
    if ( left != 0 && right > std::numeric_limits< std::size_t >::max() / left )
    {
      throw std::overflow_error( "FeedForwardNetworkConfiguration: parameter count multiplication overflow" );
    }
    return left * right;
  }

  void requireEigenIndex( std::size_t value )
  {
    if ( !std::in_range< Eigen::Index >( value ) )
    {
      throw std::overflow_error(
        "FeedForwardNetworkConfiguration: network dimension is not representable as Eigen::Index" );
    }
  }
} // namespace

std::size_t
FeedForwardNetworkConfiguration::parameterCountFromConfiguration( const FeedForwardNetworkConfiguration &configuration )
{
  std::size_t parameter_count = 0;
  std::size_t input_size = configuration.input_size;
  requireEigenIndex( input_size );

  for ( const std::size_t output_size : configuration.hidden_layers )
  {
    requireEigenIndex( output_size );
    parameter_count = checkedAdd( parameter_count, checkedMultiply( input_size, output_size ) );
    if ( configuration.use_bias )
    {
      parameter_count = checkedAdd( parameter_count, output_size );
    }
    input_size = output_size;
  }

  requireEigenIndex( configuration.output_size );
  parameter_count =
    checkedAdd( parameter_count, checkedMultiply( input_size, configuration.output_size ) );
  if ( configuration.use_bias )
  {
    parameter_count = checkedAdd( parameter_count, configuration.output_size );
  }

  requireEigenIndex( parameter_count );
  return parameter_count;
}
