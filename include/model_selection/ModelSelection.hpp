#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

/** @addtogroup model_selection_api
 * @{ */

#include <cstddef>
#include <memory>
#include <vector>
#include "ComputationCostUpperBounds.hpp"
#include "ConfigurationScoringConfiguration.hpp"
#include "ConfigurationScoringCriterion.hpp"
#include "ConfigurationSelector.hpp"
#include "ModelSelectionFunctions.hpp"
#include "ModelSelectionResult.hpp"
#include "ModelSelectionStoppingConfiguration.hpp"

/**
 * @brief Orchestrates configuration selection, scoring, retraining, and final testing.
 *
 * The object owns its ConfigurationSelector and callback bundle. Each selector
 * batch is processed completely, then returned through `tell()` in request
 * order. Budget accounting includes successful, failed, and invalid attempts;
 * a remaining budget truncates the next request before it is issued. Target and
 * no-improvement termination detected inside a batch takes effect after the
 * complete batch reaches the selector.
 *
 * InvalidConfigurationError and ConfigurationEvaluationError raised while
 * evaluating one candidate become failure records. Other exceptions propagate.
 * Successful evaluations use a finite lower-is-better score and update the
 * incumbent before feedback is sent.
 *
 * @see model_selection_chapter
 */
class ModelSelection
{
  public:
  /**
   * @brief Creates selection with validation performance as the only score term.
   *
   * Construction writes a diagnostic to `std::clog` stating that architecture
   * and computational-cost weights are disabled.
   *
   * @param[in] selector Unique ownership of the configuration generator.
   * @param[in] functions Non-empty evaluation, final-training, and final-test callbacks.
   * @param[in] stopping Distinct validated stopping alternatives in evaluation order.
   * @param[in] evaluation_batch_size Positive selector request limit; defaults to one.
   * @throws std::invalid_argument If the selector or a callback is empty.
   * @throws InvalidConfigurationError If batch size or stopping settings are
   *         invalid, or an unbounded selector has no budget criterion.
   */
  ModelSelection( std::unique_ptr< ConfigurationSelector > selector, ModelSelectionFunctions functions,
                  std::vector< ModelSelectionStoppingConfiguration > stopping,
                  std::size_t evaluation_batch_size = 1 );

  /**
   * @brief Creates selection with configured performance and cost scoring.
   * @param[in] selector Unique ownership of the configuration generator.
   * @param[in] functions Non-empty evaluation, final-training, and final-test callbacks.
   * @param[in] stopping Distinct validated stopping alternatives in evaluation order.
   * @param[in] scoring_configuration Weights, unit costs, and optional score denominators.
   * @param[in] computation_cost_upper_bounds Calculated or explicit simulation and MAC denominators.
   * @param[in] maximum_parameter_count Fallback architecture denominator when its weight is active and no explicit value is configured.
   * @param[in] evaluation_batch_size Positive selector request limit; defaults to one.
   * @throws std::invalid_argument If scoring values, denominators, selector, or callbacks are invalid.
   * @throws InvalidConfigurationError If batch size or stopping settings are
   *         invalid, or an unbounded selector has no budget criterion.
   */
  ModelSelection( std::unique_ptr< ConfigurationSelector > selector, ModelSelectionFunctions functions,
                  std::vector< ModelSelectionStoppingConfiguration > stopping,
                  ConfigurationScoringConfiguration scoring_configuration,
                  ComputationCostUpperBounds computation_cost_upper_bounds,
                  std::size_t maximum_parameter_count, std::size_t evaluation_batch_size = 1 );

  /**
   * @brief Runs selection and then trains and tests the best configuration.
   *
   * A successfully returned result contains at least one successful selection
   * evaluation. Final training uses `functions_.train_final_model`; its result
   * is passed with the best configuration to `evaluate_final_model`.
   *
   * @return Complete successful and failed history, incumbent, final outcomes, metrics, and termination reason.
   * @throws std::runtime_error If a selector returns an empty batch or selection
   *         ends without a successfully scored configuration.
   * @throws std::overflow_error If execution-metric accumulation overflows.
   * @throws std::exception Exceptions from selectors and callbacks outside the
   *         per-configuration failure categories described by the class contract.
   */
  ModelSelectionResult select();

  private:
  std::unique_ptr< ConfigurationSelector > selector_; ///< Owned stateful configuration generator.
  ModelSelectionFunctions functions_; ///< Owned validation, retraining, and test callbacks.
  std::vector< ModelSelectionStoppingConfiguration > stopping_; ///< Validated alternatives in evaluation order.
  ConfigurationScoringCriterion scoring_criterion_; ///< Lower-is-better performance and cost mapping.
  std::size_t evaluation_batch_size_; ///< Positive upper limit for one selector request.
};

/** @} */

#endif // !MODEL_SELECTION_H
