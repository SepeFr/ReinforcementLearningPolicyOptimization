#ifndef POLICY_CONFIGURATION_VALIDATOR_H
#define POLICY_CONFIGURATION_VALIDATOR_H

/** @addtogroup validation_api
 * @{ */

#include "FeedForwardNetworkConfiguration.hpp"
#include "PolicyConfiguration.hpp"

/**
 * @brief Validates feed-forward architecture and policy regularization settings.
 *
 * @see cost_validation_chapter
 */
class PolicyConfigurationValidator
{
  public:
  /** @brief Prevents construction of this stateless utility class. */
  PolicyConfigurationValidator() = delete;

  /**
   * @brief Validates layer dimensions and activation identifiers.
   * @param[in] configuration Feed-forward network configuration to inspect.
   * @throws InvalidConfigurationError If input, output, or any hidden-layer
   *         width is zero, or if an activation value is unsupported.
   */
  static void validate( const FeedForwardNetworkConfiguration &configuration );

  /**
   * @brief Validates the active policy type, architecture, and regularization.
   * @param[in] configuration Policy configuration to inspect.
   * @throws InvalidConfigurationError If the policy type or regularization type
   *         is unsupported, the coefficient is non-finite or negative, or the
   *         contained network configuration is invalid.
   */
  static void validate( const PolicyConfiguration &configuration );
};

/** @} */

#endif // !POLICY_CONFIGURATION_VALIDATOR_H
