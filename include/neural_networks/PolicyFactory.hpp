#ifndef POLICY_FACTORY_H
#define POLICY_FACTORY_H

/** @addtogroup neural_network_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <memory>
#include <stdexcept>
#include "FeedForwardNetwork.hpp"
#include "FeedForwardPolicy.hpp"
#include "ParametrizedPolicy.hpp"
#include "PolicyConfiguration.hpp"
#include "validation/InvalidConfigurationError.hpp"
#include "validation/PolicyConfigurationValidator.hpp"

/** @brief Validates PolicyConfiguration objects and creates owned policy components. */
class PolicyFactory
{
  public:
  /**
   * @brief Creates a typed parametrized policy.
   * @tparam ObservationType Observation type used by the policy interface.
   * @tparam ActionType Action type used by the policy interface.
   * @param[in] configuration Policy and network settings to validate and copy.
   * @return Unique ownership of the created policy.
   * @throws InvalidConfigurationError If the configuration violates a policy constraint.
   * @throws std::logic_error If a validated enum value has no implementation.
   */
  template< typename ObservationType = ObservationBase, typename ActionType = ActionBase >
  static std::unique_ptr< ParametrizedPolicy< ObservationType, ActionType > > create(
    const PolicyConfiguration &configuration )
  {
    PolicyConfigurationValidator::validate( configuration );

    switch ( configuration.type )
    {
      case PolicyType::FeedForwardEigen:
        return std::make_unique< FeedForwardPolicy< ObservationType, ActionType > >( configuration.network );
    }

    throw std::logic_error( "PolicyFactory: validated policy type is unsupported" );
  }

  /**
   * @brief Creates an unparameterized feed-forward network.
   * @param[in] configuration Policy configuration whose network is copied.
   * @return Unique ownership of a zero-initialized network.
   * @throws InvalidConfigurationError If the configuration violates a policy constraint.
   * @throws std::logic_error If a validated enum value has no implementation.
   */
  static std::unique_ptr< FeedForwardNetwork > create( const PolicyConfiguration &configuration )
  {
    PolicyConfigurationValidator::validate( configuration );

    switch ( configuration.type )
    {
      case PolicyType::FeedForwardEigen:
        return std::make_unique< FeedForwardNetwork >( configuration.network );
    }

    throw std::logic_error( "PolicyFactory: validated policy type is unsupported" );
  }


  /**
   * @brief Creates a feed-forward network and applies its parameters.
   * @param[in] configuration Policy configuration whose network is copied.
   * @param[in] parameters Canonical network parameter vector.
   * @return Unique ownership of the populated network.
   * @throws InvalidConfigurationError If the configuration is invalid or the
   * parameter dimension differs from its network layout.
   * @throws std::logic_error If a validated enum value has no implementation.
   */
  static std::unique_ptr< FeedForwardNetwork > create( const PolicyConfiguration &configuration,
                                                       Eigen::Ref< const Eigen::VectorXd > parameters )
  {
    PolicyConfigurationValidator::validate( configuration );

    const std::size_t expected_parameter_count =
      FeedForwardNetworkConfiguration::parameterCountFromConfiguration( configuration.network );
    if ( parameters.size() != static_cast< Eigen::Index >( expected_parameter_count ) )
    {
      throw InvalidConfigurationError( "PolicyFactory: parameter dimension does not match the policy network" );
    }

    switch ( configuration.type )
    {
      case PolicyType::FeedForwardEigen:
      {
        std::unique_ptr< FeedForwardNetwork > network = std::make_unique< FeedForwardNetwork >( configuration.network );
        network->setParameters( parameters );
        return network;
      }
    }

    throw std::logic_error( "PolicyFactory: validated policy type is unsupported" );
  }
};


/** @} */

#endif // !POLICY_FACTORY_H
