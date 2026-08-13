#include "ModelSelectionState.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "ConfigurationEvaluation.hpp"
#include "ModelConfiguration.hpp"

void ModelSelectionState::update( const ModelConfiguration &configuration, const ConfigurationEvaluation &evaluation )
{
  if ( !evaluation.selection_score.has_value() )
  {
    throw std::invalid_argument( "ModelSelectionState: selection score must be present" );
  }
  if ( !std::isfinite( *evaluation.selection_score ) )
  {
    throw std::invalid_argument( "ModelSelectionState: selection score must be finite" );
  }
  if ( !std::isfinite( evaluation.validation_mean ) )
  {
    throw std::invalid_argument( "ModelSelectionState: validation mean must be finite" );
  }
  if ( !std::isfinite( evaluation.validation_variance ) || evaluation.validation_variance < 0.0 )
  {
    throw std::invalid_argument( "ModelSelectionState: validation variance must be finite and non-negative" );
  }

  evaluations_.push_back( ModelSelectionRecord{ configuration, evaluation } );
  execution_metrics_.append( evaluation.execution_metrics );

  const std::size_t current_index = evaluations_.size() - 1;
  if ( current_index == 0 ||
       *evaluation.selection_score < *evaluations_.at( best_evaluation_index_ ).evaluation.selection_score )
  {
    best_evaluation_index_ = current_index;
  }
}

bool ModelSelectionState::empty() const { return evaluations_.empty(); }

std::size_t ModelSelectionState::evaluatedConfigurationCount() const { return evaluations_.size(); }

const ExecutionMetrics &ModelSelectionState::executionMetrics() const { return execution_metrics_; }

const ModelConfiguration &ModelSelectionState::bestConfiguration() const
{
  return evaluations_.at( best_evaluation_index_ ).configuration;
}

const ConfigurationEvaluation &ModelSelectionState::bestEvaluation() const
{
  return evaluations_.at( best_evaluation_index_ ).evaluation;
}

const std::vector< ModelSelectionRecord > &ModelSelectionState::evaluations() const { return evaluations_; }
