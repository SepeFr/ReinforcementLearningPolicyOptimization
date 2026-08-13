#include "validation/ModelConfigurationValidator.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <variant>
#include "FeedForwardNetworkConfiguration.hpp"
#include "InitializationConfiguration.hpp"
#include "ModelConfiguration.hpp"
#include "validation/InvalidConfigurationError.hpp"
#include "validation/OptimizerConfigurationValidator.hpp"
#include "validation/PolicyConfigurationValidator.hpp"

namespace
{
  bool hasSameParameterLayout( const FeedForwardNetworkConfiguration &left,
                               const FeedForwardNetworkConfiguration &right )
  {
    return left.input_size == right.input_size && left.hidden_layers == right.hidden_layers &&
      left.output_size == right.output_size && left.use_bias == right.use_bias;
  }
} // namespace

void ModelConfigurationValidator::validate( const ModelConfiguration &configuration )
{
  try
  {
    PolicyConfigurationValidator::validate( configuration.policy );

    if ( const auto *initialization = std::get_if< FeedForwardInitializationConfiguration >(
           &configuration.optimizer_configuration.initialization );
         initialization != nullptr &&
         !hasSameParameterLayout( configuration.policy.network, initialization->network ) )
    {
      throw InvalidConfigurationError(
        "ModelConfigurationValidator: feed-forward initialization layout must match the policy network" );
    }

    const std::size_t parameter_count =
      FeedForwardNetworkConfiguration::parameterCountFromConfiguration( configuration.policy.network );
    OptimizerConfigurationValidator::validate( configuration.optimizer_configuration, parameter_count );
  }
  catch ( const std::overflow_error &error )
  {
    throw InvalidConfigurationError( std::string( "ModelConfigurationValidator: " ) + error.what() );
  }
}
