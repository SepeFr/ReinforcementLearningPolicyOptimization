#ifndef EXECUTION_METRICS_H
#define EXECUTION_METRICS_H

/** @addtogroup computational_cost_api
 * @{ */

#include <cstdint>

/**
 * @brief Accumulates simulator and policy-computation counters.
 *
 * Counter updates are checked. Each mutating operation calculates all affected
 * values before committing them, so an overflow leaves the object unchanged.
 */
struct ExecutionMetrics
{
  std::uint64_t number_of_episodes = 0; ///< Number of completed episode executions.
  std::uint64_t number_of_simulation_steps = 0; ///< Total calls represented as environment steps.
  std::uint64_t number_of_policy_inferences = 0; ///< Total policy inference calls.
  std::uint64_t number_of_neural_network_macs = 0; ///< Total neural-network multiply-accumulate operations.

  /**
   * @brief Records one episode and its simulator steps.
   * @param[in] simulation_steps Number of steps executed during the episode.
   * @throws std::overflow_error If either affected counter would overflow.
   * @post On success, `number_of_episodes` is incremented by one and
   *       `simulation_steps` is added to `number_of_simulation_steps`.
   */
  void addEpisode( std::uint64_t simulation_steps );

  /**
   * @brief Records policy inferences and their neural-network work.
   * @param[in] inference_count Number of inference calls to add.
   * @param[in] macs_per_inference Multiply-accumulate operations for each call.
   * @throws std::overflow_error If the multiplication or either counter addition would overflow.
   * @post On success, inference count and \f$\text{inference_count}\times
   *       \text{macs_per_inference}\f$ MACs are added.
   */
  void addPolicyInferences( std::uint64_t inference_count, std::uint64_t macs_per_inference );

  /**
   * @brief Adds every counter from another metrics value.
   * @param[in] other Counters to accumulate.
   * @throws std::overflow_error If any counter addition would overflow.
   * @post On success, each counter equals its previous value plus the matching `other` value.
   */
  void append( const ExecutionMetrics &other );
};

/** @} */

#endif // !EXECUTION_METRICS_H
