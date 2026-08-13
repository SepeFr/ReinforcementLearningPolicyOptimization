#include "validation/PolicyConfigurationValidator.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "ActivationType.hpp"
#include "FeedForwardNetworkConfiguration.hpp"
#include "PolicyConfiguration.hpp"
#include "validation/InvalidConfigurationError.hpp"

namespace
{
  bool isKnownActivation( ActivationType type )
  {
    switch ( type )
    {
      case ActivationType::Linear:
      case ActivationType::Tanh:
      case ActivationType::Sigmoid:
      case ActivationType::ReLu:
        return true;
    }

    return false;
  }

  bool isKnownRegularization( RegularizationType type )
  {
    switch ( type )
    {
      case RegularizationType::L1_Regularization:
      case RegularizationType::L2_Regularization:
        return true;
    }

    return false;
  }
} // namespace

void PolicyConfigurationValidator::validate( const FeedForwardNetworkConfiguration &configuration )
{
  if ( configuration.input_size == 0 || configuration.output_size == 0 ||
       !std::all_of( configuration.hidden_layers.begin(), configuration.hidden_layers.end(),
                     []( std::size_t layer_size ) { return layer_size > 0; } ) ||
       !isKnownActivation( configuration.hidden_activation ) || !isKnownActivation( configuration.output_activation ) )
  {
    throw InvalidConfigurationError( "PolicyConfigurationValidator: invalid feed-forward network configuration" );
  }
}

void PolicyConfigurationValidator::validate( const PolicyConfiguration &configuration )
{
  if ( configuration.type != PolicyType::FeedForwardEigen )
  {
    throw InvalidConfigurationError( "PolicyConfigurationValidator: unsupported policy type" );
  }
  validate( configuration.network );
  if ( !isKnownRegularization( configuration.regularization_strategy ) )
  {
    throw InvalidConfigurationError( "PolicyConfigurationValidator: unsupported regularization type" );
  }
  if ( !std::isfinite( configuration.regularization_coefficient ) ||
       configuration.regularization_coefficient < 0.0 )
  {
    throw InvalidConfigurationError(
      "PolicyConfigurationValidator: regularization coefficient must be finite and non-negative" );
  }
}
