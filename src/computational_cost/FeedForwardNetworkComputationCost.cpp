#include "FeedForwardNetworkComputationCost.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include "FeedForwardNetworkConfiguration.hpp"

namespace
{
  std::uint64_t toUint64( std::size_t value )
  {
    if ( !std::in_range< std::uint64_t >( value ) )
    {
      throw std::overflow_error(
        "FeedForwardNetworkComputationCost: layer size is not representable as uint64_t" );
    }
    return static_cast< std::uint64_t >( value );
  }

  std::uint64_t checkedAdd( std::uint64_t left, std::uint64_t right )
  {
    if ( right > std::numeric_limits< std::uint64_t >::max() - left )
    {
      throw std::overflow_error( "FeedForwardNetworkComputationCost: MAC count addition overflow" );
    }
    return left + right;
  }

  std::uint64_t checkedMultiply( std::uint64_t left, std::uint64_t right )
  {
    if ( left != 0 && right > std::numeric_limits< std::uint64_t >::max() / left )
    {
      throw std::overflow_error( "FeedForwardNetworkComputationCost: MAC count multiplication overflow" );
    }
    return left * right;
  }
} // namespace

std::uint64_t
FeedForwardNetworkComputationCost::macsPerInference( const FeedForwardNetworkConfiguration &configuration )
{
  std::uint64_t macs_per_inference = 0;
  std::uint64_t input_size = toUint64( configuration.input_size );

  for ( const std::size_t output_size : configuration.hidden_layers )
  {
    const std::uint64_t represented_output_size = toUint64( output_size );
    macs_per_inference =
      checkedAdd( macs_per_inference, checkedMultiply( input_size, represented_output_size ) );
    input_size = represented_output_size;
  }

  macs_per_inference =
    checkedAdd( macs_per_inference, checkedMultiply( input_size, toUint64( configuration.output_size ) ) );
  return macs_per_inference;
}
