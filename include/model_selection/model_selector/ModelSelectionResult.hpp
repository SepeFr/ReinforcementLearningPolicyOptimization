#ifndef MODEL_SELECTION_RESULT_H
#define MODEL_SELECTION_RESULT_H

/** @addtogroup model_selection_api
 * @{ */

/** @file ModelSelectionResult.hpp @brief Selection history, failure records, and final outcomes. */

#include <cstddef>
#include <optional>
#include <string>
#include <vector>
#include "ExecutionMetrics.hpp"
#include "ObjectiveEvaluation.hpp"
#include "OptimizerResult.hpp"
#include "ConfigurationEvaluation.hpp"
#include "ModelConfiguration.hpp"
#include "ModelSelectionState.hpp"
#include "OptimizationTerminationReason.hpp"

/** @brief Classifies configuration attempts excluded from successful model-selection state. */
enum class ModelSelectionFailureKind
{
  InvalidConfiguration, ///< ModelConfigurationValidator rejected the candidate.
  EvaluationFailure, ///< Training or validation raised a general ConfigurationEvaluationError.
  NonFiniteValidationStatistics, ///< Validation mean or variance was unusable.
  NonFiniteSelectionScore ///< Performance and cost terms produced an unusable score.
};

/** @brief Preserves diagnostic data for one failed configuration attempt. */
struct ModelSelectionFailureRecord
{
  ModelConfiguration configuration; ///< Attempted configuration copy.
  ModelSelectionFailureKind kind; ///< Stable machine-readable failure category.
  std::string message; ///< Diagnostic text from validation, evaluation, or scoring.
  std::optional< ConfigurationEvaluation > evaluation; ///< Partial evaluation when one was returned before rejection.
};

/**
 * @brief Returns selection history, final training and test outcomes, and execution cost.
 *
 * `selection_execution_metrics` contains successful selection evaluations.
 * `total_execution_metrics` adds final training and test metrics.
 */
struct ModelSelectionResult
{
  ModelConfiguration best_configuration; ///< Configuration with the lowest finite selection score; earliest on ties.
  ConfigurationEvaluation best_evaluation; ///< Validation result and score of `best_configuration`.
  std::vector< ModelSelectionRecord > evaluated_configurations; ///< Successful records in evaluation order.
  std::size_t attempted_configuration_count = 0; ///< Successful, failed, and invalid attempts consumed during selection.
  std::vector< ModelSelectionFailureRecord > failed_configurations; ///< Failed attempts in evaluation order.
  OptimizerResult final_training_result; ///< Retraining outcome for the best configuration.
  ObjectiveEvaluation test_evaluation; ///< Natural-direction evaluation on the final test scenarios.
  ExecutionMetrics selection_execution_metrics; ///< Sum of metrics from successful selection evaluations.
  ExecutionMetrics total_execution_metrics; ///< Selection metrics plus final training and test metrics.
  OptimizationTerminationReason termination_reason = OptimizationTerminationReason::None; ///< First stopping reason or selector exhaustion.
};

/** @} */

#endif // !MODEL_SELECTION_RESULT_H
