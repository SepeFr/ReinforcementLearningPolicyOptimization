#include "ModelSelection.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include "ComputationCostUpperBounds.hpp"
#include "ConfigurationEvaluation.hpp"
#include "ConfigurationEvaluationError.hpp"
#include "ConfigurationScoringConfiguration.hpp"
#include "ConfigurationScoringCriterion.hpp"
#include "ConfigurationSelector.hpp"
#include "ExecutionMetrics.hpp"
#include "ModelConfiguration.hpp"
#include "ModelSelectionFunctions.hpp"
#include "ModelSelectionResult.hpp"
#include "ModelSelectionState.hpp"
#include "ModelSelectionStoppingConfiguration.hpp"
#include "ModelSelectionStoppingCriterion.hpp"
#include "ObjectiveEvaluation.hpp"
#include "OptimizationTerminationReason.hpp"
#include "OptimizerResult.hpp"
#include "validation/InvalidConfigurationError.hpp"
#include "validation/ModelConfigurationValidator.hpp"

namespace
{
  void validateStoppingConfigurations( const std::vector< ModelSelectionStoppingConfiguration > &configurations )
  {
    std::array< bool, std::variant_size_v< ModelSelectionStoppingConfiguration > > configured_types{};

    for ( const ModelSelectionStoppingConfiguration &configuration : configurations )
    {
      if ( configuration.valueless_by_exception() || configured_types.at( configuration.index() ) )
      {
        throw InvalidConfigurationError( "ModelSelection: invalid or duplicate stopping criterion" );
      }
      configured_types.at( configuration.index() ) = true;

      const bool valid = std::visit(
        []( const auto &selected_configuration )
        {
          using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

          if constexpr ( std::is_same_v< ConfigurationType, ModelSelectionBudgetCriterionConfiguration > )
          {
            return selected_configuration.maximum_iterations > 0;
          }
          else if constexpr (
            std::is_same_v< ConfigurationType, ModelSelectionNoImprovementCriterionConfiguration > )
          {
            return selected_configuration.maximum_iterations_without_improvement > 0 &&
              std::isfinite( selected_configuration.threshold ) && selected_configuration.threshold >= 0.0;
          }
          else
          {
            return std::isfinite( selected_configuration.target_score );
          }
        },
        configuration );

      if ( !valid )
      {
        throw InvalidConfigurationError( "ModelSelection: invalid stopping criterion configuration" );
      }
    }
  }

  std::optional< std::size_t >
  remainingConfigurationBudget( const std::vector< ModelSelectionStoppingConfiguration > &stopping,
                                std::size_t evaluated_configuration_count )
  {
    for ( const ModelSelectionStoppingConfiguration &configuration : stopping )
    {
      if ( const auto *budget = std::get_if< ModelSelectionBudgetCriterionConfiguration >( &configuration ) )
      {
        if ( evaluated_configuration_count >= budget->maximum_iterations )
        {
          return 0;
        }
        return budget->maximum_iterations - evaluated_configuration_count;
      }
    }

    return std::nullopt;
  }

  bool hasStoppingBudget( const std::vector< ModelSelectionStoppingConfiguration > &stopping )
  {
    return std::any_of(
      stopping.begin(), stopping.end(), []( const ModelSelectionStoppingConfiguration &configuration )
      { return std::holds_alternative< ModelSelectionBudgetCriterionConfiguration >( configuration ); } );
  }

  ModelSelectionFailureKind failureKind( ConfigurationEvaluationErrorKind kind )
  {
    switch ( kind )
    {
      case ConfigurationEvaluationErrorKind::NonFiniteValidationStatistics:
        return ModelSelectionFailureKind::NonFiniteValidationStatistics;
      case ConfigurationEvaluationErrorKind::NonFiniteSelectionScore:
        return ModelSelectionFailureKind::NonFiniteSelectionScore;
      case ConfigurationEvaluationErrorKind::EvaluationFailure:
        return ModelSelectionFailureKind::EvaluationFailure;
    }

    return ModelSelectionFailureKind::EvaluationFailure;
  }

  ConfigurationScoringCriterion
  makeScoringCriterion( ConfigurationScoringConfiguration configuration,
                        ComputationCostUpperBounds computation_cost_upper_bounds,
                        std::size_t calculated_maximum_parameter_count )
  {
    const std::size_t maximum_parameter_count = configuration.architecture_complexity_weight == 0.0 ? 1 :
      configuration.maximum_parameter_count.value_or( calculated_maximum_parameter_count );

    return ConfigurationScoringCriterion( std::move( configuration ), computation_cost_upper_bounds,
                                          maximum_parameter_count );
  }
} // namespace

ModelSelection::ModelSelection( std::unique_ptr< ConfigurationSelector > selector, ModelSelectionFunctions functions,
                                std::vector< ModelSelectionStoppingConfiguration > stopping,
                                std::size_t evaluation_batch_size ) :
    ModelSelection( std::move( selector ), std::move( functions ), std::move( stopping ),
                    ConfigurationScoringConfiguration{}, ComputationCostUpperBounds{}, 1, evaluation_batch_size )
{
  std::clog << "ModelSelection warning: legacy construction disables architecture and computational cost "
               "regularization\n";
}

ModelSelection::ModelSelection( std::unique_ptr< ConfigurationSelector > selector, ModelSelectionFunctions functions,
                                std::vector< ModelSelectionStoppingConfiguration > stopping,
                                ConfigurationScoringConfiguration scoring_configuration,
                                ComputationCostUpperBounds computation_cost_upper_bounds,
                                std::size_t maximum_parameter_count, std::size_t evaluation_batch_size ) :
    selector_( std::move( selector ) ), functions_( std::move( functions ) ), stopping_( std::move( stopping ) ),
    scoring_criterion_( makeScoringCriterion( std::move( scoring_configuration ), computation_cost_upper_bounds,
                                              maximum_parameter_count ) ),
    evaluation_batch_size_( evaluation_batch_size )
{
  if ( !selector_ )
  {
    throw std::invalid_argument( "ModelSelection: selector cannot be null" );
  }
  if ( !functions_.evaluate_configuration )
  {
    throw std::invalid_argument( "ModelSelection: configuration evaluation function cannot be empty" );
  }
  if ( !functions_.train_final_model )
  {
    throw std::invalid_argument( "ModelSelection: final training function cannot be empty" );
  }
  if ( !functions_.evaluate_final_model )
  {
    throw std::invalid_argument( "ModelSelection: final evaluation function cannot be empty" );
  }

  if ( evaluation_batch_size_ == 0 )
  {
    throw InvalidConfigurationError( "ModelSelection: evaluation batch size must be greater than zero" );
  }

  validateStoppingConfigurations( stopping_ );
  if ( selector_->requiresStoppingBudget() && !hasStoppingBudget( stopping_ ) )
  {
    throw InvalidConfigurationError(
      "ModelSelection: this configuration selector requires a budget stopping criterion" );
  }
}

ModelSelectionResult ModelSelection::select()
{
  ModelSelectionStoppingCriterion stopping_criterion( stopping_ );
  ModelSelectionState state;
  std::size_t attempted_configuration_count = 0;
  std::vector< ModelSelectionFailureRecord > failed_configurations;
  OptimizationTerminationReason termination_reason =
    OptimizationTerminationReason::ConfigurationSelectorExhausted;

  // Every candidate configuration is evaluated under the same
  // maximum black-box evaluation budget.
  while ( selector_->hasNext() )
  {
    std::size_t maximum_batch_size = evaluation_batch_size_;
    const std::optional< std::size_t > remaining_budget =
      remainingConfigurationBudget( stopping_, attempted_configuration_count );

    if ( remaining_budget.has_value() )
    {
      if ( *remaining_budget == 0 )
      {
        termination_reason = OptimizationTerminationReason::MaximumIterations;
        break;
      }
      maximum_batch_size = std::min( maximum_batch_size, *remaining_budget );
    }

    const std::vector< ModelConfiguration > configurations = selector_->ask( maximum_batch_size );
    if ( configurations.empty() )
    {
      throw std::runtime_error( "ModelSelection: selector returned an empty configuration batch" );
    }

    std::vector< ConfigurationFeedback > feedback;
    feedback.reserve( configurations.size() );
    std::optional< OptimizationTerminationReason > stopping_reason;

    // Selector batches are atomic: target and no-improvement are honored after complete feedback reaches tell(),
    // so they may overshoot only by the configurations remaining in this already-issued batch.
    for ( const ModelConfiguration &configuration : configurations )
    {
      // Invalid configurations still consume a model-selection budget slot, but no evaluation resources.
      attempted_configuration_count++;
      std::optional< ConfigurationEvaluation > evaluation;
      try
      {
        ModelConfigurationValidator::validate( configuration );
        evaluation = functions_.evaluate_configuration( configuration );

        if ( !std::isfinite( evaluation->validation_mean ) ||
             !std::isfinite( evaluation->validation_variance ) || evaluation->validation_variance < 0.0 )
        {
          constexpr const char *message =
            "ModelSelection: validation statistics must be finite and variance must be non-negative";
          feedback.push_back( ConfigurationFeedback{ *evaluation, ConfigurationEvaluationStatus::Failed } );
          failed_configurations.push_back(
            ModelSelectionFailureRecord{ configuration, ModelSelectionFailureKind::NonFiniteValidationStatistics,
                                         message, evaluation } );
        }
        else
        {
          const double selection_score =
            scoring_criterion_.score( configuration, *evaluation, functions_.optimization_direction );
          if ( !std::isfinite( selection_score ) )
          {
            constexpr const char *message = "ModelSelection: selection score must be finite";
            feedback.push_back( ConfigurationFeedback{ *evaluation, ConfigurationEvaluationStatus::Failed } );
            failed_configurations.push_back(
              ModelSelectionFailureRecord{ configuration, ModelSelectionFailureKind::NonFiniteSelectionScore,
                                           message, evaluation } );
          }
          else
          {
            evaluation->selection_score = selection_score;
            state.update( configuration, *evaluation );
            feedback.push_back( ConfigurationFeedback{ *evaluation, ConfigurationEvaluationStatus::Succeeded } );
          }
        }
      }
      catch ( const InvalidConfigurationError &error )
      {
        feedback.push_back( ConfigurationFeedback{ evaluation.value_or( ConfigurationEvaluation{} ),
                                                   ConfigurationEvaluationStatus::Failed } );
        failed_configurations.push_back(
          ModelSelectionFailureRecord{ configuration, ModelSelectionFailureKind::InvalidConfiguration,
                                       error.what(), evaluation } );
      }
      catch ( const ConfigurationEvaluationError &error )
      {
        feedback.push_back( ConfigurationFeedback{ evaluation.value_or( ConfigurationEvaluation{} ),
                                                   ConfigurationEvaluationStatus::Failed } );
        failed_configurations.push_back(
          ModelSelectionFailureRecord{ configuration, failureKind( error.kind() ), error.what(), evaluation } );
      }

      if ( !stopping_reason.has_value() )
      {
        stopping_reason = stopping_criterion.stoppingReason( feedback.back() );
      }
    }

    selector_->tell( feedback );

    if ( stopping_reason.has_value() )
    {
      termination_reason = *stopping_reason;
      break;
    }
  }

  if ( state.empty() )
  {
    throw std::runtime_error(
      "ModelSelection: selection finished without a successfully evaluated configuration" );
  }

  OptimizerResult final_training_result = functions_.train_final_model( state.bestConfiguration() );
  ObjectiveEvaluation test_evaluation =
    functions_.evaluate_final_model( state.bestConfiguration(), final_training_result );

  ExecutionMetrics selection_execution_metrics = state.executionMetrics();
  ExecutionMetrics total_execution_metrics = selection_execution_metrics;
  total_execution_metrics.append( final_training_result.execution_metrics );
  total_execution_metrics.append( test_evaluation.executionMetrics() );

  return ModelSelectionResult{ state.bestConfiguration(),
                               state.bestEvaluation(),
                               state.evaluations(),
                               attempted_configuration_count,
                               std::move( failed_configurations ),
                               std::move( final_training_result ),
                               std::move( test_evaluation ),
                               selection_execution_metrics,
                               total_execution_metrics,
                               termination_reason };
}
