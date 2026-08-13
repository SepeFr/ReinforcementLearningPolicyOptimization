#include "ConfigurationScoringCriterion.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include "ComputationCostUpperBounds.hpp"
#include "ConfigurationEvaluationError.hpp"
#include "ConfigurationEvaluation.hpp"
#include "ConfigurationScoringConfiguration.hpp"
#include "FeedForwardNetworkConfiguration.hpp"
#include "ModelConfiguration.hpp"

namespace
{
  void validateNonNegative( double value, const char *message )
  {
    if ( !std::isfinite( value ) || value < 0.0 )
    {
      throw std::invalid_argument( message );
    }
  }

  void validatePositive( double value, const char *message )
  {
    if ( !std::isfinite( value ) || value <= 0.0 )
    {
      throw std::invalid_argument( message );
    }
  }

  void warnIfUpperBoundExceeded( const char *quantity, double normalized_value )
  {
    if ( normalized_value > 1.0 )
    {
      std::clog << "ConfigurationScoringCriterion warning: normalized " << quantity << " is " << normalized_value
                << " and exceeds its declared upper bound\n";
    }
  }
} // namespace

ConfigurationScoringCriterion::ConfigurationScoringCriterion( ConfigurationScoringConfiguration configuration,
                                                               ComputationCostUpperBounds upper_bounds,
                                                               std::size_t maximum_parameter_count ) :
    configuration_( std::move( configuration ) ), upper_bounds_( upper_bounds ),
    maximum_parameter_count_( maximum_parameter_count )
{
  validateNonNegative( configuration_.reset_cost,
                       "ConfigurationScoringCriterion: reset cost must be finite and non-negative" );
  validateNonNegative( configuration_.simulation_step_cost,
                       "ConfigurationScoringCriterion: simulation step cost must be finite and non-negative" );
  validateNonNegative(
    configuration_.architecture_complexity_weight,
    "ConfigurationScoringCriterion: architecture complexity weight must be finite and non-negative" );
  validateNonNegative( configuration_.simulation_cost_weight,
                       "ConfigurationScoringCriterion: simulation cost weight must be finite and non-negative" );
  validateNonNegative(
    configuration_.neural_network_cost_weight,
    "ConfigurationScoringCriterion: neural network cost weight must be finite and non-negative" );

  if ( configuration_.architecture_complexity_weight > 0.0 && maximum_parameter_count_ == 0 )
  {
    throw std::invalid_argument( "ConfigurationScoringCriterion: maximum parameter count must be positive" );
  }
  if ( configuration_.simulation_cost_weight > 0.0 )
  {
    validatePositive( upper_bounds_.maximum_simulation_cost,
                      "ConfigurationScoringCriterion: maximum simulation cost must be finite and positive" );
  }
  if ( configuration_.neural_network_cost_weight > 0.0 )
  {
    validatePositive( upper_bounds_.maximum_neural_network_cost,
                      "ConfigurationScoringCriterion: maximum neural network cost must be finite and positive" );
  }
}

double ConfigurationScoringCriterion::score( const ModelConfiguration &model_configuration,
                                              const ConfigurationEvaluation &evaluation,
                                              OptimizationDirection direction ) const
{
  if ( !std::isfinite( evaluation.validation_mean ) )
  {
    throw std::invalid_argument( "ConfigurationScoringCriterion: validation mean must be finite" );
  }
  if ( !std::isfinite( evaluation.validation_variance ) || evaluation.validation_variance < 0.0 )
  {
    throw std::invalid_argument( "ConfigurationScoringCriterion: validation variance must be finite and non-negative" );
  }

  double architecture_penalty = 0.0;
  if ( configuration_.architecture_complexity_weight > 0.0 )
  {
    const std::size_t parameter_count =
      FeedForwardNetworkConfiguration::parameterCountFromConfiguration( model_configuration.policy.network );
    const double normalized_parameter_count =
      static_cast< double >( parameter_count ) / static_cast< double >( maximum_parameter_count_ );
    warnIfUpperBoundExceeded( "parameter count", normalized_parameter_count );
    architecture_penalty = configuration_.architecture_complexity_weight * normalized_parameter_count;
  }

  double simulation_penalty = 0.0;
  if ( configuration_.simulation_cost_weight > 0.0 )
  {
    const double simulation_cost =
      configuration_.reset_cost * static_cast< double >( evaluation.execution_metrics.number_of_episodes ) +
      configuration_.simulation_step_cost *
        static_cast< double >( evaluation.execution_metrics.number_of_simulation_steps );
    const double normalized_simulation_cost = simulation_cost / upper_bounds_.maximum_simulation_cost;
    warnIfUpperBoundExceeded( "simulation cost", normalized_simulation_cost );
    simulation_penalty = configuration_.simulation_cost_weight * normalized_simulation_cost;
  }

  double neural_network_penalty = 0.0;
  if ( configuration_.neural_network_cost_weight > 0.0 )
  {
    const double normalized_neural_network_cost =
      static_cast< double >( evaluation.execution_metrics.number_of_neural_network_macs ) /
      upper_bounds_.maximum_neural_network_cost;
    warnIfUpperBoundExceeded( "neural network cost", normalized_neural_network_cost );
    neural_network_penalty = configuration_.neural_network_cost_weight * normalized_neural_network_cost;
  }

  double performance_score = evaluation.validation_mean;
  if ( direction == OptimizationDirection::Maximize )
  {
    performance_score = -performance_score;
  }

  const double selection_score =
    performance_score + architecture_penalty + simulation_penalty + neural_network_penalty;
  if ( !std::isfinite( selection_score ) )
  {
    throw ConfigurationEvaluationError(
      "ConfigurationScoringCriterion: selection score must be finite",
      ConfigurationEvaluationErrorKind::NonFiniteSelectionScore );
  }

  return selection_score;
}
