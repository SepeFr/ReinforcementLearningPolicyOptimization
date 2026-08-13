#ifndef CONFIGURATION_SCORING_CONFIGURATION_H
#define CONFIGURATION_SCORING_CONFIGURATION_H

/** @addtogroup computational_cost_api
 * @{ */

#include <cstddef>
#include <optional>

/**
 * @brief Configures performance and computational-cost contributions to a selection score.
 *
 * All costs and weights must be finite and non-negative. A zero weight disables
 * its component. The resulting score always follows a minimization convention.
 *
 * @see ConfigurationScoringCriterion
 */
struct ConfigurationScoringConfiguration
{
  double reset_cost = 0.0; ///< Cost units assigned to each completed environment reset/episode.
  double simulation_step_cost = 1.0; ///< Cost units assigned to each simulator step.
  double architecture_complexity_weight = 0.0; ///< Weight of normalized neural-network parameter count.
  double simulation_cost_weight = 0.0; ///< Weight of normalized reset and simulator-step cost.
  double neural_network_cost_weight = 0.0; ///< Weight of normalized neural-network MAC count.

  std::optional< std::size_t > maximum_parameter_count; ///< Positive architecture denominator; may be derived from the search space.
  std::optional< double > maximum_simulation_cost; ///< Finite positive simulation-cost denominator, or empty for automatic calculation.
  std::optional< double > maximum_neural_network_cost; ///< Finite positive MAC-count denominator, or empty for automatic calculation.
};

/** @} */

#endif // !CONFIGURATION_SCORING_CONFIGURATION_H
