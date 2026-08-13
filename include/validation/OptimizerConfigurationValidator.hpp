#ifndef OPTIMIZER_CONFIGURATION_VALIDATOR_H
#define OPTIMIZER_CONFIGURATION_VALIDATOR_H

/** @addtogroup validation_api
 * @{ */

#include <cstddef>
#include "OptimizerConfiguration.hpp"

/**
 * @brief Validates optimizer-wide and method-specific configuration invariants.
 *
 * The checks cover parameter bounds, initialization cardinality, stopping
 * criteria, and the active OptimizerMethodConfiguration alternative. Dimension
 * requirements are evaluated against the supplied model parameter count.
 *
 * @see cost_validation_chapter
 */
class OptimizerConfigurationValidator
{
  public:
  /** @brief Prevents construction of this stateless utility class. */
  OptimizerConfigurationValidator() = delete;

  /**
   * @brief Validates an optimizer configuration for a concrete dimension.
   * @param[in] configuration Optimizer settings and selected method configuration.
   * @param[in] parameter_count Number of scalar decision variables; must be positive.
   * @throws InvalidConfigurationError If the dimension is invalid or any
   *         initializer, stopping criterion, bound, or method constraint fails.
   */
  static void validate( const OptimizerConfiguration &configuration, std::size_t parameter_count );
};

/** @} */

#endif // !OPTIMIZER_CONFIGURATION_VALIDATOR_H
