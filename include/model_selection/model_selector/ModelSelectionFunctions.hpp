#ifndef MODEL_SELECTION_FUNCTIONS_H
#define MODEL_SELECTION_FUNCTIONS_H

/** @addtogroup model_selection_api
 * @{ */

/** @file ModelSelectionFunctions.hpp @brief Model-selection callbacks and their standard protocol adapter. */

#include <functional>
#include <utility>
#include <vector>
#include "ConfigurationEvaluation.hpp"
#include "BlackBoxProblem.hpp"
#include "ModelConfiguration.hpp"
#include "ObjectiveEvaluation.hpp"
#include "OptimizerResult.hpp"

/**
 * @brief Supplies model selection with validation, final training, and test operations.
 *
 * The callables are owned by value. Exceptions they throw propagate from
 * ModelSelection except for InvalidConfigurationError and
 * ConfigurationEvaluationError raised during configuration evaluation.
 */
struct ModelSelectionFunctions
{
  std::function< ConfigurationEvaluation( const ModelConfiguration & ) > evaluate_configuration; ///< Trains and validates one candidate configuration.
  std::function< OptimizerResult( const ModelConfiguration & ) > train_final_model; ///< Retrains the selected configuration for final testing.
  std::function< ObjectiveEvaluation( const ModelConfiguration &, const OptimizerResult & ) > evaluate_final_model; ///< Evaluates final trained parameters on the test scenarios.
  OptimizationDirection optimization_direction = OptimizationDirection::Minimize; ///< Natural direction converted by ConfigurationScoringCriterion.
};

/**
 * @brief Creates the standard validation, retraining, and test callbacks.
 * @tparam ValidationProtocol Protocol type exposing compatible static `evaluate()` and `train()` operations.
 * @tparam ValidationConfiguration Holdout or K-fold configuration type.
 * @tparam Scenario Test and training scenario value type.
 *
 * The validation configuration and test scenarios are captured by value. Its
 * environment remains a borrowed reference. Final training concatenates the
 * configured training scenarios followed by validation scenarios. Final test
 * evaluation uses `validation_runs_per_scenario`. The returned natural direction
 * is `OptimizationDirection::Maximize`.
 *
 * @param[in] validation_configuration Protocol settings captured by the callbacks.
 * @param[in] test_scenarios Test scenarios moved into the final-evaluation callback.
 * @return Fully populated callbacks for ModelSelection.
 * @pre The referenced environment outlives every callback invocation.
 */
template< typename ValidationProtocol, typename ValidationConfiguration, typename Scenario >
ModelSelectionFunctions makeModelSelectionFunctions( ValidationConfiguration validation_configuration,
                                                     std::vector< Scenario > test_scenarios )
{
  return ModelSelectionFunctions{
    [validation_configuration]( const ModelConfiguration &model_configuration )
    { return ValidationProtocol::evaluate( validation_configuration, model_configuration ); },
    [validation_configuration]( const ModelConfiguration &model_configuration )
    {
      std::vector< Scenario > final_training_scenarios = validation_configuration.training_scenarios;
      final_training_scenarios.insert( final_training_scenarios.end(),
                                       validation_configuration.validation_scenarios.begin(),
                                       validation_configuration.validation_scenarios.end() );

      return ValidationProtocol::train( model_configuration, final_training_scenarios,
                                        validation_configuration.environment.get(),
                                        validation_configuration.training_runs_per_scenario );
    },
    [validation_configuration, test_scenarios = std::move( test_scenarios )](
      const ModelConfiguration &model_configuration, const OptimizerResult &training_result )
    {
      return ValidationProtocol::evaluate( model_configuration, training_result, test_scenarios,
                                           validation_configuration.environment.get(),
                                           validation_configuration.validation_runs_per_scenario );
    },
    OptimizationDirection::Maximize };
}

/** @} */

#endif // !MODEL_SELECTION_FUNCTIONS_H
