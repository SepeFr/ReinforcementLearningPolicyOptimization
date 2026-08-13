#ifndef OPTIMIZATION_STOPPING_STRATEGY_H
#define OPTIMIZATION_STOPPING_STRATEGY_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
#include "CandidateEvaluation.hpp"
#include "OptimizationTerminationReason.hpp"
#include "StoppingConfiguration.hpp"

/** @brief Interface for one stateful external optimization stopping test. */
class OptimizationStoppingStrategy
{
  public:
  /** @brief Enables destruction through the strategy interface. */
  virtual ~OptimizationStoppingStrategy() = default;

  /**
   * @brief Restricts a requested batch before evaluation.
   * @param[in] requested_count Candidate count returned by ask().
   * @return Number of leading candidates that fit this criterion's allowance.
   */
  virtual std::size_t allowedEvaluationCount( std::size_t requested_count ) const = 0;
  /**
   * @brief Updates criterion state and tests its stopping condition.
   * @param[in] current_evaluations Most recently evaluated candidate batch.
   * @param[in] best_candidate Current method incumbent in minimization space.
   * @param[in] method_iterations Completed logical iteration count.
   * @return `true` when the run must stop for this criterion.
   */
  virtual bool shouldStop( const std::vector< CandidateEvaluation > &current_evaluations,
                           const CandidateEvaluation &best_candidate,
                           std::size_t method_iterations ) = 0;
  /** @brief Identifies this criterion in the final result. @return Associated termination reason. */
  virtual OptimizationTerminationReason terminationReason() const = 0;
};

/** @brief Tests the completed logical-iteration count against a fixed limit. */
class MaximumIterationsStoppingCriterion final : public OptimizationStoppingStrategy
{
  public:
  /** @brief Stores an iteration limit. @param[in] configuration Validated limit. */
  explicit MaximumIterationsStoppingCriterion( MaximumIterationsStoppingConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /** @copydoc OptimizationStoppingStrategy::allowedEvaluationCount() */
  std::size_t allowedEvaluationCount( std::size_t requested_count ) const override { return requested_count; }

  /** @copydoc OptimizationStoppingStrategy::terminationReason() */
  OptimizationTerminationReason terminationReason() const override
  {
    return OptimizationTerminationReason::MaximumIterations;
  }

  /**
   * @brief Tests whether the completed logical-iteration limit has been reached.
   * @param[in] method_iterations Completed logical iterations.
   * @return `true` when @p method_iterations is at least maximum_iterations.
   */
  bool shouldStop( const std::vector< CandidateEvaluation > &, const CandidateEvaluation &,
                   std::size_t method_iterations ) override
  {
    return method_iterations >= configuration_.maximum_iterations;
  }

  private:
  /** Copied iteration limit. */
  MaximumIterationsStoppingConfiguration configuration_;
};

/** @brief Enforces a cumulative attempted-candidate budget. */
class MaximumEvaluationsStoppingCriterion final : public OptimizationStoppingStrategy
{
  public:
  /** @brief Stores an evaluation budget with a zero initial count. @param[in] configuration Validated budget. */
  explicit MaximumEvaluationsStoppingCriterion( MaximumEvaluationsStoppingConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /**
   * @brief Limits a batch to the remaining budget.
   * @param[in] requested_count Candidate count returned by ask().
   * @return Minimum of @p requested_count and the remaining allowance.
   */
  std::size_t allowedEvaluationCount( std::size_t requested_count ) const override
  {
    const std::size_t remaining_count = evaluation_count_ < configuration_.maximum_evaluations
      ? configuration_.maximum_evaluations - evaluation_count_
      : 0;
    return std::min( requested_count, remaining_count );
  }

  /** @copydoc OptimizationStoppingStrategy::terminationReason() */
  OptimizationTerminationReason terminationReason() const override
  {
    return OptimizationTerminationReason::MaximumEvaluations;
  }

  /**
   * @brief Accounts for the current batch and tests budget exhaustion.
   * @param[in] current_evaluations Attempted candidates in this batch.
   * @return `true` when the cumulative count reaches the maximum.
   */
  bool shouldStop( const std::vector< CandidateEvaluation > &current_evaluations,
                   const CandidateEvaluation &, std::size_t ) override
  {
    evaluation_count_ += current_evaluations.size();
    return evaluation_count_ >= configuration_.maximum_evaluations;
  }

  private:
  /** Copied attempted-candidate limit. */
  MaximumEvaluationsStoppingConfiguration configuration_;
  /** Attempted candidates accounted through shouldStop(). */
  std::size_t evaluation_count_ = 0;
};

/** @brief Counts completed iterations lacking a strict threshold improvement. */
class NoImprovementStoppingCriterion final : public OptimizationStoppingStrategy
{
  public:
  /** @brief Stores the span and improvement threshold. @param[in] configuration Validated settings. */
  explicit NoImprovementStoppingCriterion( NoImprovementStoppingConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /** @copydoc OptimizationStoppingStrategy::allowedEvaluationCount() */
  std::size_t allowedEvaluationCount( std::size_t requested_count ) const override { return requested_count; }

  /** @copydoc OptimizationStoppingStrategy::terminationReason() */
  OptimizationTerminationReason terminationReason() const override
  {
    return OptimizationTerminationReason::NoImprovement;
  }

  /**
   * @brief Processes each new logical iteration once.
   * @param[in] best_candidate Current successful incumbent when available.
   * @param[in] method_iterations Completed logical iteration count.
   * @return `true` after the configured number of iterations without a decrease
   * strictly greater than threshold.
   */
  bool shouldStop( const std::vector< CandidateEvaluation > &,
                   const CandidateEvaluation &best_candidate, std::size_t method_iterations ) override
  {
    if ( method_iterations == 0 )
    {
      if ( !best_value_.has_value() && best_candidate.status == CandidateEvaluationStatus::Succeeded )
      {
        best_value_ = best_candidate.meanValue();
      }
      return false;
    }

    if ( method_iterations <= last_processed_method_iteration_ )
    {
      return false;
    }
    last_processed_method_iteration_ = method_iterations;

    if ( best_candidate.status != CandidateEvaluationStatus::Succeeded )
    {
      iterations_without_improvement_++;
      return iterations_without_improvement_ >= configuration_.maximum_iterations_without_improvement;
    }

    const double current_best_value = best_candidate.meanValue();

    if ( !best_value_.has_value() )
    {
      best_value_ = current_best_value;
      iterations_without_improvement_ = 0;
      return iterations_without_improvement_ >= configuration_.maximum_iterations_without_improvement;
    }

    const bool improved = *best_value_ - current_best_value > configuration_.threshold;

    if ( improved )
    {
      best_value_ = current_best_value;
      iterations_without_improvement_ = 0;
    }
    else
    {
      iterations_without_improvement_++;
    }

    return iterations_without_improvement_ >= configuration_.maximum_iterations_without_improvement;
  }

  private:
  /** Copied span and threshold. */
  NoImprovementStoppingConfiguration configuration_;
  /** Consecutive processed iterations without sufficient decrease. */
  std::size_t iterations_without_improvement_ = 0;
  /** Largest logical iteration index already processed. */
  std::size_t last_processed_method_iteration_ = 0;
  /** Incumbent mean used as the improvement baseline. */
  std::optional< double > best_value_;
};

/** @brief Stops when a successful minimization-space incumbent reaches a target. */
class TargetValueStoppingCriterion final : public OptimizationStoppingStrategy
{
  public:
  /** @brief Stores a direction-normalized target. @param[in] configuration Target in minimization space. */
  explicit TargetValueStoppingCriterion( TargetValueStoppingConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /** @copydoc OptimizationStoppingStrategy::allowedEvaluationCount() */
  std::size_t allowedEvaluationCount( std::size_t requested_count ) const override { return requested_count; }

  /** @copydoc OptimizationStoppingStrategy::terminationReason() */
  OptimizationTerminationReason terminationReason() const override
  {
    return OptimizationTerminationReason::TargetReached;
  }

  /**
   * @brief Tests a successful incumbent against the target.
   * @param[in] best_candidate Current minimization-space incumbent.
   * @return `true` when a successful mean is at most target_value.
   */
  bool shouldStop( const std::vector< CandidateEvaluation > &,
                   const CandidateEvaluation &best_candidate, std::size_t ) override
  {
    return best_candidate.status == CandidateEvaluationStatus::Succeeded &&
      best_candidate.meanValue() <= configuration_.target_value;
  }

  private:
  /** Copied direction-normalized target. */
  TargetValueStoppingConfiguration configuration_;
};

/** @} */

#endif // !OPTIMIZATION_STOPPING_STRATEGY_H
