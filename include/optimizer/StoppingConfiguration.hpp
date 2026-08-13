#ifndef STOPPING_CONFIGURATION_H
#define STOPPING_CONFIGURATION_H

/** @addtogroup optimizer_core_api
 * @{ */

/** @file StoppingConfiguration.hpp @brief External optimizer stopping configuration alternatives. */

#include <cstddef>
#include <variant>

struct MaximumIterationsStoppingConfiguration;
struct MaximumEvaluationsStoppingConfiguration;
struct NoImprovementStoppingConfiguration;
struct TargetValueStoppingConfiguration;

/** @brief Variant selecting one external optimizer stopping criterion. */
using StoppingConfiguration =
  std::variant< MaximumIterationsStoppingConfiguration, MaximumEvaluationsStoppingConfiguration,
                NoImprovementStoppingConfiguration, TargetValueStoppingConfiguration >;

/** @brief Stops after a limit of completed logical method iterations. */
struct MaximumIterationsStoppingConfiguration
{
  std::size_t maximum_iterations; ///< Inclusive iteration limit; must be positive.
};

/** @brief Stops at an attempted-candidate budget. */
struct MaximumEvaluationsStoppingConfiguration
{
  std::size_t maximum_evaluations; ///< Maximum candidates attempted, including rejected candidates; must be positive.
};

/** @brief Stops after consecutive completed iterations without sufficient improvement. */
struct NoImprovementStoppingConfiguration
{
  std::size_t maximum_iterations_without_improvement; ///< Positive consecutive-iteration limit.
  double threshold; ///< Finite non-negative decrease required to reset the counter; comparison is strict.
};

/** @brief Stops when the incumbent reaches a target in the problem's public direction. */
struct TargetValueStoppingConfiguration
{
  double target_value; ///< Finite target; internally negated for a maximizing problem.
};

/** @} */

#endif // !STOPPING_CONFIGURATION_H
