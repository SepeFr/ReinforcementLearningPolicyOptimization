#ifndef VALIDATION_PROTOCOL_H
#define VALIDATION_PROTOCOL_H

/** @addtogroup model_selection_api
 * @{ */

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include "ConfigurationEvaluationError.hpp"
#include "ConfigurationEvaluation.hpp"
#include "ExecutionMetrics.hpp"
#include "ModelConfiguration.hpp"
#include "ObjectiveEvaluation.hpp"
#include "OptimizerFactory.hpp"
#include "OptimizerResult.hpp"
#include "PolicyOptimizationProblem.hpp"
#include "ValidationProtocolConfiguration.hpp"

/**
 * @brief Trains one model configuration and evaluates it on a holdout set.
 * @tparam ScenarioType Scenario value accepted by the environment reset contract.
 * @tparam ObservationType Observation returned by the environment.
 * @tparam ActionType Action returned by the policy.
 * @tparam EnvironmentType Environment implementation shared across episodes.
 *
 * Training constructs a PolicyOptimizationProblem and an Optimizer. Validation
 * evaluates the trained best parameters in the objective's natural direction
 * with policy regularization excluded. The protocol owns temporary problems and
 * optimizers; it borrows the environment stored in its configuration.
 *
 * @see KFoldCrossValidationProtocol
 * @see model_selection_chapter
 */
template< typename ScenarioType, typename ObservationType, typename ActionType, typename EnvironmentType >
class ValidationProtocol
{
  using Scenario = ScenarioType; ///< Scenario type used by public protocol operations.
  using Environment = EnvironmentType; ///< Borrowed environment implementation type.
  using ProblemType = PolicyOptimizationProblem< ScenarioType, ObservationType, ActionType, EnvironmentType >; ///< Training and evaluation objective type.

  public:
  /**
   * @brief Runs training followed by fixed-set holdout validation.
   *
   * Training metrics and validation metrics are added. Validation samples must
   * be non-empty and finite. The returned variance is the square of
   * ObjectiveEvaluation::standardDeviation().
   *
   * @param[in] validation_configuration Scenario sets, replica counts, and borrowed environment.
   * @param[in] model_configuration Policy and optimizer settings used for training.
   * @return Natural-direction validation mean and population variance, combined metrics, and an empty selection score.
   * @throws std::invalid_argument If either scenario set is empty or either replica count is zero.
   * @throws ConfigurationEvaluationError If validation produces no samples,
   *         non-finite samples, or invalid aggregate statistics.
   * @throws std::overflow_error If an execution metric counter overflows.
   */
  static ConfigurationEvaluation
  evaluate( const ValidationProtocolConfiguration< Scenario, Environment > &validation_configuration,
            const ModelConfiguration &model_configuration )
  {
    validateConfiguration( validation_configuration );
    Environment &environment = validation_configuration.environment.get();

    const OptimizerResult training_result = train( model_configuration, validation_configuration.training_scenarios,
                                                   environment, validation_configuration.training_runs_per_scenario );

    ObjectiveEvaluation validation_evaluation =
      evaluate( model_configuration, training_result, validation_configuration.validation_scenarios, environment,
                validation_configuration.validation_runs_per_scenario );

    ExecutionMetrics execution_metrics = training_result.execution_metrics;
    execution_metrics.append( validation_evaluation.executionMetrics() );

    if ( validation_evaluation.sampleCount() == 0 )
    {
      throw ConfigurationEvaluationError(
        "ValidationProtocol: validation evaluation must contain at least one sample" );
    }
    if ( !validation_evaluation.allFinite() )
    {
      throw ConfigurationEvaluationError(
        "ValidationProtocol: validation evaluation must contain only finite samples",
        ConfigurationEvaluationErrorKind::NonFiniteValidationStatistics );
    }

    const double validation_mean = validation_evaluation.meanValue();
    const double standard_deviation = validation_evaluation.standardDeviation();
    const double validation_variance = standard_deviation * standard_deviation;

    if ( !std::isfinite( validation_mean ) || !std::isfinite( validation_variance ) || validation_variance < 0.0 )
    {
      throw ConfigurationEvaluationError(
        "ValidationProtocol: validation mean and variance must be finite and valid",
        ConfigurationEvaluationErrorKind::NonFiniteValidationStatistics );
    }

    return ConfigurationEvaluation{ validation_mean, validation_variance, execution_metrics, std::nullopt };
  }

  /**
   * @brief Optimizes policy parameters on a supplied training scenario set.
   *
   * PolicyOptimizationProblem applies the regularization configured in
   * `model_configuration.policy` during training.
   *
   * @param[in] model_configuration Policy and optimizer settings.
   * @param[in] training_scenarios Scenarios evaluated for every candidate.
   * @param[in,out] environment Borrowed environment reset and stepped by episodes.
   * @param[in] number_of_runs Replica episodes per scenario and candidate.
   * @return Optimizer result containing the best parameter vector and training metrics.
   * @throws std::invalid_argument If problem construction or optimizer configuration is invalid.
   * @throws std::runtime_error If optimization cannot produce a successful incumbent.
   */
  static OptimizerResult train( const ModelConfiguration &model_configuration,
                                const std::vector< Scenario > &training_scenarios, Environment &environment,
                                std::size_t number_of_runs )
  {
    auto training_problem =
      std::make_unique< ProblemType >( model_configuration.policy, training_scenarios, number_of_runs, environment );

    std::unique_ptr< Optimizer > optimizer =
      OptimizerFactory::create( std::move( training_problem ), model_configuration.optimizer_configuration );

    return optimizer->optimize();
  }

  /**
   * @brief Evaluates trained parameters on scenarios without regularization.
   * @param[in] model_configuration Policy architecture used to reconstruct the trained policy.
   * @param[in] training_result Source of `best_candidate.parameters`.
   * @param[in] evaluation_scenarios Scenarios evaluated in stored order.
   * @param[in,out] environment Borrowed environment reset and stepped by episodes.
   * @param[in] number_of_runs Replica episodes per scenario.
   * @return Objective samples and execution metrics in the problem's natural direction.
   * @throws std::invalid_argument If the scenario set, replica count, policy configuration, or parameter vector is invalid.
   */
  static ObjectiveEvaluation evaluate( const ModelConfiguration &model_configuration,
                                       const OptimizerResult &training_result,
                                       const std::vector< Scenario > &evaluation_scenarios, Environment &environment,
                                       std::size_t number_of_runs )
  {
    ProblemType evaluation_problem( model_configuration.policy, evaluation_scenarios, number_of_runs, environment,
                                    RegularizationApplication::Exclude );

    return evaluation_problem.evaluate( training_result.best_candidate.parameters );
  }

  private:
  /** @brief Prevents construction of this stateless protocol utility. */
  ValidationProtocol() = delete;

  /**
   * @brief Checks holdout scenario and replica cardinalities.
   * @param[in] validation_configuration Configuration to inspect.
   * @throws std::invalid_argument If either scenario set is empty or either replica count is zero.
   */
  static void
  validateConfiguration( const ValidationProtocolConfiguration< Scenario, Environment > &validation_configuration )
  {
    if ( validation_configuration.training_scenarios.empty() )
    {
      throw std::invalid_argument( "ValidationProtocol: training scenarios cannot be empty" );
    }
    if ( validation_configuration.validation_scenarios.empty() )
    {
      throw std::invalid_argument( "ValidationProtocol: validation scenarios cannot be empty" );
    }
    if ( validation_configuration.training_runs_per_scenario == 0 )
    {
      throw std::invalid_argument( "ValidationProtocol: training runs per scenario must be greater than zero" );
    }
    if ( validation_configuration.validation_runs_per_scenario == 0 )
    {
      throw std::invalid_argument( "ValidationProtocol: validation runs per scenario must be greater than zero" );
    }
  }
};

/** @} */

#endif // !VALIDATION_PROTOCOL_H
