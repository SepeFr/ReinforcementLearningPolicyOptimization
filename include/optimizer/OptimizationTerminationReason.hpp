#ifndef OPTIMIZATION_TERMINATION_REASON_H
#define OPTIMIZATION_TERMINATION_REASON_H

/** @addtogroup optimizer_core_api
 * @{ */

/** @file OptimizationTerminationReason.hpp @brief Shared optimizer and model-selection termination reasons. */

/** @brief Condition that ended an optimizer or model-selection run. */
enum class OptimizationTerminationReason
{
  None,                           ///< No termination condition has been recorded.
  MethodConverged,                ///< The optimization method's internal test converged.
  MaximumIterations,              ///< A logical-iteration limit was reached.
  MaximumEvaluations,             ///< An attempted-candidate budget was exhausted.
  NoImprovement,                  ///< The incumbent failed to improve for the configured span.
  TargetReached,                  ///< The configured objective or score target was met.
  ConfigurationSelectorExhausted  ///< A finite configuration selector has no remaining candidates.
};

/** @} */

#endif // !OPTIMIZATION_TERMINATION_REASON_H
