#ifndef OPTIMIZATION_STOPPING_CRITERION_H
#define OPTIMIZATION_STOPPING_CRITERION_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include "BlackBoxProblem.hpp"
#include "CandidateEvaluation.hpp"
#include "OptimizationStoppingStrategy.hpp"
#include "OptimizationTerminationReason.hpp"
#include "StoppingConfiguration.hpp"

/**
 * @brief Combines at most one stopping strategy of each configuration type.
 *
 * Strategies are evaluated in configuration order. The first satisfied
 * strategy supplies the termination reason. Batch allowance is the minimum
 * returned by all installed strategies.
 */
class OptimizationStoppingCriterion
{
  public:
  /**
   * @brief Constructs a composite containing one strategy.
   * @param[in] configuration Strategy settings.
   * @param[in] direction Natural problem direction used to normalize targets.
   */
  explicit OptimizationStoppingCriterion(
    StoppingConfiguration configuration,
    OptimizationDirection direction = OptimizationDirection::Minimize );
  /**
   * @brief Constructs strategies in the supplied precedence order.
   * @param[in] configurations Strategy settings; each variant alternative may appear once.
   * @param[in] direction Natural problem direction used to normalize targets.
   * @throws std::invalid_argument If a configuration type is duplicated.
   */
  explicit OptimizationStoppingCriterion(
    const std::vector< StoppingConfiguration > &configurations,
    OptimizationDirection direction = OptimizationDirection::Minimize );

  /**
   * @brief Computes how many leading candidates fit every installed allowance.
   * @param[in] requested_count Candidate count returned by a method.
   * @return Minimum allowance, or @p requested_count when no criterion is installed.
   */
  std::size_t allowedEvaluationCount( std::size_t requested_count ) const;
  /**
   * @brief Identifies the first criterion that would truncate a batch.
   * @param[in] requested_count Candidate count returned by a method.
   * @return Its termination reason, or OptimizationTerminationReason::None.
   */
  OptimizationTerminationReason evaluationLimitReason( std::size_t requested_count ) const;
  /**
   * @brief Updates strategies in precedence order and returns the first match.
   * @param[in] current_evaluations Most recently processed batch.
   * @param[in] best_candidate Current minimization-space method incumbent.
   * @param[in] method_iterations Completed logical iteration count.
   * @return First satisfied reason, or std::nullopt.
   */
  std::optional< OptimizationTerminationReason >
  stoppingReason( const std::vector< CandidateEvaluation > &current_evaluations,
                  const CandidateEvaluation &best_candidate,
                  std::size_t method_iterations );

  private:
  /** One ownership slot per StoppingConfiguration variant alternative. */
  using CriterionSlots =
    std::array< std::unique_ptr< OptimizationStoppingStrategy >, std::variant_size_v< StoppingConfiguration > >;

  /**
   * @brief Creates and inserts one uniquely typed criterion.
   * @param[in] configuration Settings whose variant index selects the slot.
   * @param[in] direction Natural direction used for target normalization.
   * @throws std::invalid_argument If the selected slot is already occupied.
   */
  void addCriterion( const StoppingConfiguration &configuration, OptimizationDirection direction );

  /** Strategies owned in fixed variant-index slots. */
  CriterionSlots criteria_{};
  /** Occupied slot indices in caller-specified precedence order. */
  std::vector< std::size_t > criterion_order_;
};

/** @} */

#endif // !OPTIMIZATION_STOPPING_CRITERION_H
