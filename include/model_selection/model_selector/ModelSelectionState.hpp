#ifndef MODEL_SELECTION_STATE_H
#define MODEL_SELECTION_STATE_H

/** @addtogroup model_selection_api
 * @{ */

#include <cstddef>
#include <vector>
#include "ConfigurationEvaluation.hpp"
#include "ExecutionMetrics.hpp"
#include "ModelConfiguration.hpp"

/** @brief Stores one successfully scored configuration in evaluation order. */
struct ModelSelectionRecord
{
  ModelConfiguration configuration; ///< Evaluated configuration copy.
  ConfigurationEvaluation evaluation; ///< Finite statistics, metrics, and engaged selection score.
};

/**
 * @brief Tracks successful evaluations, their cost, and the best score.
 *
 * Lower selection scores are better. Equal scores retain the earliest record.
 * Failed attempts are recorded by ModelSelectionResult and never enter this state.
 */
class ModelSelectionState
{
  public:
  /**
   * @brief Appends a successful evaluation and updates the incumbent.
   * @param[in] configuration Evaluated model configuration.
   * @param[in] evaluation Evaluation with finite mean, finite non-negative variance, and finite score.
   * @throws std::invalid_argument If statistics or selection score violate the required contract.
   * @throws std::overflow_error If accumulating execution metrics overflows.
   */
  void update( const ModelConfiguration &configuration, const ConfigurationEvaluation &evaluation );

  /** @brief Reports whether any successful record exists. @return `true` before the first successful update. */
  bool empty() const;
  /** @brief Returns the number of successful evaluations. @return `evaluations().size()`. */
  std::size_t evaluatedConfigurationCount() const;
  /** @brief Returns metrics accumulated from successful evaluations. @return State-owned metrics reference. */
  const ExecutionMetrics &executionMetrics() const;

  /**
   * @brief Returns the configuration with the lowest selection score.
   * @return Reference into the state-owned evaluation records.
   * @throws std::out_of_range If the state is empty.
   */
  const ModelConfiguration &bestConfiguration() const;
  /**
   * @brief Returns the evaluation associated with `bestConfiguration()`.
   * @return Reference into the state-owned evaluation records.
   * @throws std::out_of_range If the state is empty.
   */
  const ConfigurationEvaluation &bestEvaluation() const;
  /** @brief Returns all successful records in evaluation order. @return State-owned vector reference. */
  const std::vector< ModelSelectionRecord > &evaluations() const;

  private:
  std::vector< ModelSelectionRecord > evaluations_; ///< Successful records in request order.
  std::size_t best_evaluation_index_ = 0; ///< Position of the first record attaining the lowest score.
  ExecutionMetrics execution_metrics_; ///< Sum of metrics in `evaluations_`.
};

/** @} */

#endif // !MODEL_SELECTION_STATE_H
