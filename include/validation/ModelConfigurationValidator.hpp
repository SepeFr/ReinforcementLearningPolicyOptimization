#ifndef MODEL_CONFIGURATION_VALIDATOR_H
#define MODEL_CONFIGURATION_VALIDATOR_H

/** @addtogroup validation_api
 * @{ */

#include "ModelConfiguration.hpp"

/**
 * @brief Validates the cross-component invariants of a model configuration.
 *
 * The validator checks the policy, derives its parameter count, verifies any
 * feed-forward initialization layout against the policy architecture, and then
 * validates the optimizer for that parameter count.
 *
 * @see cost_validation_chapter
 */
class ModelConfigurationValidator
{
  public:
  /** @brief Prevents construction of this stateless utility class. */
  ModelConfigurationValidator() = delete;

  /**
   * @brief Validates a complete policy-and-optimizer configuration.
   * @param[in] configuration Model configuration to inspect without modifying it.
   * @throws InvalidConfigurationError If the policy, initialization layout,
   *         derived parameter count, or optimizer configuration is invalid.
   */
  static void validate( const ModelConfiguration &configuration );
};

/** @} */

#endif // !MODEL_CONFIGURATION_VALIDATOR_H
