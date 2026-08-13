#ifndef K_FOLD_CROSS_VALIDATION_PROTOCOL_H
#define K_FOLD_CROSS_VALIDATION_PROTOCOL_H

/** @addtogroup model_selection_api
 * @{ */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "ConfigurationEvaluationError.hpp"
#include "ExecutionMetrics.hpp"
#include "ValidationProtocol.hpp"

/**
 * @brief Evaluates a model configuration through reproducible K-fold validation.
 * @tparam ScenarioType Scenario value accepted by the environment reset contract.
 * @tparam ObservationType Observation returned by the environment.
 * @tparam ActionType Action returned by the policy.
 * @tparam EnvironmentType Environment implementation shared across folds.
 *
 * The original training and validation vectors are concatenated in that order,
 * permuted from `random_seed`, and partitioned into contiguous validation folds.
 * Earlier folds receive one extra scenario when cardinality is not divisible by
 * `fold_count`. Every fold trains a fresh optimizer on its complement.
 *
 * @see model_selection_chapter
 */
template< typename ScenarioType, typename ObservationType, typename ActionType, typename EnvironmentType >
class KFoldCrossValidationProtocol final
    : public ValidationProtocol< ScenarioType, ObservationType, ActionType, EnvironmentType >
{
  using Base = ValidationProtocol< ScenarioType, ObservationType, ActionType, EnvironmentType >; ///< Holdout operations reused per fold.
  using Scenario = ScenarioType; ///< Scenario value type.
  using Environment = EnvironmentType; ///< Borrowed environment type.

  public:
  // Expose the base protocol's documented low-level evaluation overload.
  using Base::evaluate;
  // Expose the base protocol's documented training operation.
  using Base::train;

  /**
   * @brief Runs all folds and aggregates validation statistics by sample count.
   *
   * For fold \f$k\f$, let \f$n_k\f$ be its number of scenarios times
   * `validation_runs_per_scenario`, with mean \f$\widehat J_k(c)\f$ and variance
   * \f$v_k\f$. The combined statistics are
   * \f[
   * \overline J(c)=\frac{\sum_k n_k\widehat J_k(c)}{\sum_k n_k},\qquad
   * v=\frac{\sum_k n_k\left(v_k+\widehat J_k(c)^2\right)}{\sum_k n_k}-\overline J(c)^2.
   * \f]
   * A negative `v` caused by the computed second moment is clamped to zero.
   * Metrics are accumulated across training and validation for every fold.
   *
   * @param[in] validation_configuration Combined scenario pool, fold count, seed, replicas, and environment.
   * @param[in] model_configuration Policy and optimizer settings reused for every fresh fold training.
   * @return Weighted natural-direction mean and variance, all-fold metrics, and an empty selection score.
   * @throws std::invalid_argument If fold count is below two, exceeds combined
   *         scenario count, or inherited scenario/replica constraints fail.
   * @throws ConfigurationEvaluationError If a fold or aggregate statistic is non-finite or otherwise invalid.
   * @throws std::overflow_error If accumulated execution metrics overflow.
   */
  static ConfigurationEvaluation
  evaluate( const KFoldCrossValidationConfiguration< Scenario, Environment > &validation_configuration,
            const ModelConfiguration &model_configuration )
  {
    if ( validation_configuration.fold_count < 2 )
    {
      throw std::invalid_argument( "KFoldCrossValidationProtocol: fold count must be at least two" );
    }

    std::vector< Scenario > scenarios;
    scenarios.reserve( validation_configuration.training_scenarios.size() +
                       validation_configuration.validation_scenarios.size() );
    scenarios.insert( scenarios.end(), validation_configuration.training_scenarios.begin(),
                      validation_configuration.training_scenarios.end() );
    scenarios.insert( scenarios.end(), validation_configuration.validation_scenarios.begin(),
                      validation_configuration.validation_scenarios.end() );

    if ( scenarios.size() < validation_configuration.fold_count )
    {
      throw std::invalid_argument( "KFoldCrossValidationProtocol: scenario count must be at least the fold count" );
    }

    // A local generator makes the fold permutation reproducible for each call.
    std::mt19937_64 generator( validation_configuration.random_seed );
    std::vector< std::size_t > shuffled_indices( scenarios.size() );
    std::iota( shuffled_indices.begin(), shuffled_indices.end(), 0 );
    std::shuffle( shuffled_indices.begin(), shuffled_indices.end(), generator );

    // Aggregate through weighted first and second moments.
    double weighted_mean_sum = 0.0;
    double weighted_second_moment_sum = 0.0;
    std::size_t validation_sample_count = 0;
    ExecutionMetrics execution_metrics;
    const std::size_t base_fold_size = scenarios.size() / validation_configuration.fold_count;
    const std::size_t remaining_scenarios = scenarios.size() % validation_configuration.fold_count;
    std::size_t validation_begin = 0;

    for ( std::size_t fold = 0; fold < validation_configuration.fold_count; ++fold )
    {
      const std::size_t validation_size = base_fold_size + ( fold < remaining_scenarios ? 1 : 0 );
      const std::size_t validation_end = validation_begin + validation_size;

      std::vector< Scenario > training_scenarios;
      std::vector< Scenario > validation_scenarios;
      training_scenarios.reserve( scenarios.size() - validation_size );
      validation_scenarios.reserve( validation_size );

      for ( std::size_t position = 0; position < shuffled_indices.size(); ++position )
      {
        const Scenario &scenario = scenarios.at( shuffled_indices.at( position ) );
        if ( position >= validation_begin && position < validation_end )
        {
          validation_scenarios.push_back( scenario );
        }
        else
        {
          training_scenarios.push_back( scenario );
        }
      }

      ValidationProtocolConfiguration< Scenario, Environment > fold_configuration{
        std::move( training_scenarios ), std::move( validation_scenarios ), validation_configuration.environment,
        validation_configuration.training_runs_per_scenario, validation_configuration.validation_runs_per_scenario };

      const ConfigurationEvaluation fold_evaluation = Base::evaluate( fold_configuration, model_configuration );

      if ( !std::isfinite( fold_evaluation.validation_mean ) ||
           !std::isfinite( fold_evaluation.validation_variance ) || fold_evaluation.validation_variance < 0.0 )
      {
        throw ConfigurationEvaluationError(
          "KFoldCrossValidationProtocol: fold mean and variance must be finite and valid",
          ConfigurationEvaluationErrorKind::NonFiniteValidationStatistics );
      }

      const std::size_t fold_sample_count = validation_size * validation_configuration.validation_runs_per_scenario;
      const double weighted_mean = static_cast< double >( fold_sample_count ) * fold_evaluation.validation_mean;
      const double weighted_second_moment = static_cast< double >( fold_sample_count ) *
        ( fold_evaluation.validation_variance + fold_evaluation.validation_mean * fold_evaluation.validation_mean );

      const double updated_weighted_mean_sum = weighted_mean_sum + weighted_mean;
      const double updated_weighted_second_moment_sum = weighted_second_moment_sum + weighted_second_moment;
      if ( !std::isfinite( weighted_mean ) || !std::isfinite( weighted_second_moment ) ||
           !std::isfinite( updated_weighted_mean_sum ) || !std::isfinite( updated_weighted_second_moment_sum ) )
      {
        throw ConfigurationEvaluationError(
          "KFoldCrossValidationProtocol: fold aggregation must remain finite",
          ConfigurationEvaluationErrorKind::NonFiniteValidationStatistics );
      }

      weighted_mean_sum = updated_weighted_mean_sum;
      weighted_second_moment_sum = updated_weighted_second_moment_sum;
      validation_sample_count += fold_sample_count;
      execution_metrics.append( fold_evaluation.execution_metrics );
      validation_begin = validation_end;
    }

    const double validation_mean = weighted_mean_sum / static_cast< double >( validation_sample_count );
    const double raw_validation_variance =
      weighted_second_moment_sum / static_cast< double >( validation_sample_count ) -
      validation_mean * validation_mean;

    if ( !std::isfinite( validation_mean ) || !std::isfinite( raw_validation_variance ) )
    {
      throw ConfigurationEvaluationError(
        "KFoldCrossValidationProtocol: aggregated mean and variance must be finite",
        ConfigurationEvaluationErrorKind::NonFiniteValidationStatistics );
    }

    const double validation_variance = raw_validation_variance < 0.0 ? 0.0 : raw_validation_variance;
    if ( !std::isfinite( validation_variance ) || validation_variance < 0.0 )
    {
      throw ConfigurationEvaluationError(
        "KFoldCrossValidationProtocol: aggregated variance must be valid",
        ConfigurationEvaluationErrorKind::NonFiniteValidationStatistics );
    }

    return ConfigurationEvaluation{ validation_mean, validation_variance, execution_metrics, std::nullopt };
  }

  private:
  /** @brief Prevents construction of this stateless protocol utility. */
  KFoldCrossValidationProtocol() = delete;
};

/** @} */

#endif // !K_FOLD_CROSS_VALIDATION_PROTOCOL_H
