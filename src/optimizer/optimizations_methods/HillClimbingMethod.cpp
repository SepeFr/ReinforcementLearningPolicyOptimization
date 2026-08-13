#include "optimizations_methods/HillClimbingMethod.hpp"
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <optional>
#include <stdexcept>
#include <vector>
#include "CandidateEvaluation.hpp"
#include "ConfigurationEvaluationError.hpp"
#include "ObjectiveEvaluation.hpp"

std::vector< Eigen::VectorXd > HillClimbingMethod::ask()
{
  if ( not has_current_candidate_ )
  {
    current_candidate_ =
      CandidateEvaluation{ initial_parameters_strategy_->generateInitialParameters(), ObjectiveEvaluation{} };
    return { current_candidate_.parameters };
  }

  std::vector< Eigen::VectorXd > neighbors =
    neighborhood_->generateNeighbors( current_candidate_.parameters, hyperparameters_.number_of_neighbors );

  for ( Eigen::VectorXd &neighbor : neighbors )
  {
    const Eigen::VectorXd displacement = neighbor - current_candidate_.parameters;
    neighbor = current_candidate_.parameters + hyperparameters_.neighborhood_scale * displacement;
  }

  return neighbors;
}


OptimizerMethodUpdate HillClimbingMethod::tellImpl( const std::vector< CandidateEvaluation > &evaluations )
{
  if ( not has_current_candidate_ )
  {
    if ( evaluations.at( 0 ).status != CandidateEvaluationStatus::Succeeded )
    {
      throw ConfigurationEvaluationError( "HillClimbingMethod: initial candidate evaluation failed" );
    }

    current_candidate_ = evaluations.at( 0 );
    has_current_candidate_ = true;
    return {};
  }

  const std::optional< CandidateEvaluation > best_candidate = extractBestCandidate( evaluations );
  if ( !best_candidate.has_value() )
  {
    return {};
  }

  const double best_candidate_value = best_candidate->meanValue();
  const double current_value = current_candidate_.meanValue();
  const bool is_better = best_candidate_value < current_value;
  const bool is_equal = best_candidate_value == current_value;

  if ( is_better || ( hyperparameters_.accept_equal_candidates && is_equal ) )
  {
    current_candidate_ = *best_candidate;
  }

  return {};
}


void HillClimbingMethod::reset()
{
  OptimizerMethod::reset();
  neighborhood_->reset();
  current_candidate_ = CandidateEvaluation{ Eigen::VectorXd{}, ObjectiveEvaluation{} };
  has_current_candidate_ = false;
}
