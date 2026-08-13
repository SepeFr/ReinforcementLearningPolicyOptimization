#ifndef MODEL_SELECTION_STOPPING_STRATEGY_H
#define MODEL_SELECTION_STOPPING_STRATEGY_H

/** @addtogroup model_selection_api
 * @{ */

#include <cstddef>
#include <optional>
#include <utility>
#include "ConfigurationSelector.hpp"
#include "ModelSelectionStoppingConfiguration.hpp"
#include "OptimizationTerminationReason.hpp"

/**
 * @brief Stateful interface for one model-selection stopping rule.
 *
 * Successful feedback supplied to score-based strategies must contain an
 * engaged finite `selection_score`.
 */
class ModelSelectionStoppingStrategy
{
  public:
  /** @brief Destroys a stopping strategy through its interface. */
  virtual ~ModelSelectionStoppingStrategy() = default;

  /**
   * @brief Updates strategy state from one attempted configuration.
   * @param[in] current_feedback Ordered outcome for the current attempt.
   * @return `true` when the strategy's condition is satisfied after the update.
   */
  virtual bool shouldStop( const ConfigurationFeedback &current_feedback ) = 0;
  /** @brief Returns the termination reason represented by this strategy. @return Stable reason value. */
  virtual OptimizationTerminationReason terminationReason() const = 0;
};

/** @brief Counts every attempted configuration against a fixed budget. */
class ModelSelectionBudgetCriterion final : public ModelSelectionStoppingStrategy
{
  public:
  /**
   * @brief Creates an attempt counter starting at zero.
   * @param[in] configuration Positive maximum iteration count.
   */
  explicit ModelSelectionBudgetCriterion( ModelSelectionBudgetCriterionConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /**
   * @brief Increments the attempt counter and tests it against the configured limit.
   * @return `true` when the updated count reaches `maximum_iterations`.
   */
  bool shouldStop( const ConfigurationFeedback & ) override
  {
    iteration_count_++;
    return iteration_count_ >= configuration_.maximum_iterations;
  }

  /** @copydoc ModelSelectionStoppingStrategy::terminationReason */
  OptimizationTerminationReason terminationReason() const override
  {
    return OptimizationTerminationReason::MaximumIterations;
  }

  private:
  ModelSelectionBudgetCriterionConfiguration configuration_; ///< Fixed positive attempt limit.
  std::size_t iteration_count_ = 0; ///< Feedback items processed so far.
};

/** @brief Tracks strict lower-is-better improvement across successful evaluations. */
class ModelSelectionNoImprovementCriterion final : public ModelSelectionStoppingStrategy
{
  public:
  /**
   * @brief Creates an empty incumbent and a zero non-improvement count.
   * @param[in] configuration Positive count limit and finite non-negative threshold.
   */
  explicit ModelSelectionNoImprovementCriterion( ModelSelectionNoImprovementCriterionConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /**
   * @brief Updates the best score and non-improvement run.
   *
   * Failed feedback leaves the state unchanged. A successful score improves the
   * incumbent exactly when `best_score - current_score > threshold`.
   *
   * @param[in] current_feedback Outcome to process.
   * @return `true` when the non-improvement count reaches its configured limit.
   * @pre Successful feedback contains an engaged finite selection score.
   */
  bool shouldStop( const ConfigurationFeedback &current_feedback ) override
  {
    if ( current_feedback.status != ConfigurationEvaluationStatus::Succeeded )
    {
      return false;
    }

    const ConfigurationEvaluation &current_evaluation = current_feedback.evaluation;
    const double current_score = *current_evaluation.selection_score;

    if ( !best_score_.has_value() )
    {
      best_score_ = current_score;
      iterations_without_improvement_ = 0;
      return iterations_without_improvement_ >= configuration_.maximum_iterations_without_improvement;
    }

    const bool improved = *best_score_ - current_score > configuration_.threshold;

    if ( improved )
    {
      best_score_ = current_score;
      iterations_without_improvement_ = 0;
    }
    else
    {
      iterations_without_improvement_++;
    }

    return iterations_without_improvement_ >= configuration_.maximum_iterations_without_improvement;
  }

  /** @copydoc ModelSelectionStoppingStrategy::terminationReason */
  OptimizationTerminationReason terminationReason() const override
  {
    return OptimizationTerminationReason::NoImprovement;
  }

  private:
  ModelSelectionNoImprovementCriterionConfiguration configuration_; ///< Count limit and strict improvement threshold.
  std::size_t iterations_without_improvement_ = 0; ///< Consecutive successful evaluations without sufficient decrease.
  std::optional< double > best_score_; ///< Lowest successful score observed so far.
};

/** @brief Stops when a successful score is at or below a fixed target. */
class ModelSelectionTargetScoreCriterion final : public ModelSelectionStoppingStrategy
{
  public:
  /**
   * @brief Creates a target-score strategy.
   * @param[in] configuration Finite inclusive target.
   */
  explicit ModelSelectionTargetScoreCriterion( ModelSelectionTargetScoreCriterionConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
  }

  /**
   * @brief Tests successful feedback against the target.
   * @param[in] current_feedback Outcome to inspect.
   * @return `true` for a successful score satisfying `score <= target_score`.
   * @pre Successful feedback contains an engaged finite selection score.
   */
  bool shouldStop( const ConfigurationFeedback &current_feedback ) override
  {
    return current_feedback.status == ConfigurationEvaluationStatus::Succeeded &&
      *current_feedback.evaluation.selection_score <= configuration_.target_score;
  }

  /** @copydoc ModelSelectionStoppingStrategy::terminationReason */
  OptimizationTerminationReason terminationReason() const override
  {
    return OptimizationTerminationReason::TargetReached;
  }

  private:
  ModelSelectionTargetScoreCriterionConfiguration configuration_; ///< Fixed inclusive lower-is-better target.
};

/** @} */

#endif // !MODEL_SELECTION_STOPPING_STRATEGY_H
