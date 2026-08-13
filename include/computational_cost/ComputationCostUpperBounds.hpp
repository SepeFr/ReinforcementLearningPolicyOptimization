#ifndef COMPUTATION_COST_UPPER_BOUNDS_H
#define COMPUTATION_COST_UPPER_BOUNDS_H

/**
 * @defgroup computational_cost_api Computational Cost
 * @brief Execution metrics, cost models, upper bounds, and configuration scoring.
 * @{
 */

#include <cstddef>
#include <optional>

/**
 * @brief Inputs used to derive conservative normalization bounds.
 *
 * Each value is required only when the corresponding cost component is active
 * and its bound is not supplied directly in ConfigurationScoringConfiguration.
 *
 * @see ComputationCostUpperBoundsCalculator
 * @see cost_validation_chapter
 */
struct ComputationCostUpperBoundInputs
{
  std::optional< std::size_t > maximum_black_box_evaluations; ///< Maximum objective evaluations per training run.
  std::optional< std::size_t > maximum_episode_steps; ///< Maximum simulator steps in one episode.
  std::optional< std::size_t > maximum_macs_per_inference; ///< Maximum multiply-accumulate operations per policy inference.
};

/**
 * @brief Positive denominators used to normalize measured execution costs.
 *
 * A value of `1.0` is retained for an inactive score component. Active
 * components receive either a caller-provided bound or a bound calculated from
 * the validation protocol and ComputationCostUpperBoundInputs.
 */
struct ComputationCostUpperBounds
{
  double maximum_simulation_cost = 1.0; ///< Upper bound for reset and simulator-step cost units.
  double maximum_neural_network_cost = 1.0; ///< Upper bound for neural-network multiply-accumulate operations.
};

/** @} */

#endif // !COMPUTATION_COST_UPPER_BOUNDS_H
