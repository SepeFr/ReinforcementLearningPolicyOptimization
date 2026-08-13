#include "optimizations_methods/simulated_annealing/SimulatedAnnealingMethod.hpp"
#include <cmath>
#include <optional>
#include <stdexcept>
#include "ConfigurationEvaluationError.hpp"

std::vector< Eigen::VectorXd > SimulatedAnnealingMethod::ask()
{
  if ( not has_current_candidate_ )
  {
    current_candidate_ =
      CandidateEvaluation{ initial_parameters_strategy_->generateInitialParameters(), ObjectiveEvaluation{} };
    return { current_candidate_.parameters };
  }


  Eigen::VectorXd generated_neighbor = neighborhood_->generateNeighbor( current_candidate_.parameters );
  Eigen::VectorXd displacement = generated_neighbor - current_candidate_.parameters;
  return { current_candidate_.parameters + hyperparameters_.neighborhood_scale * displacement };
}


OptimizerMethodUpdate SimulatedAnnealingMethod::tellImpl( const std::vector< CandidateEvaluation > &evaluations )
{
  if ( not has_current_candidate_ )
  {
    if ( evaluations.at( 0 ).status != CandidateEvaluationStatus::Succeeded )
    {
      throw ConfigurationEvaluationError( "SimulatedAnnealingMethod: initial candidate evaluation failed" );
    }

    current_candidate_ = evaluations.at( 0 );
    has_current_candidate_ = true;
    return {};
  }

  const std::optional< CandidateEvaluation > best_candidate = extractBestCandidate( evaluations );
  if ( !best_candidate.has_value() )
  {
    // A failed proposal still consumes one annealing step.
    time_++;
    return {};
  }

  const double best_candidate_value = best_candidate->meanValue();
  const double current_value = current_candidate_.meanValue();
  const bool is_better = best_candidate_value < current_value;

  if ( is_better )
  {
    current_candidate_ = *best_candidate;
  }
  else
  {
    const double delta_energy = best_candidate_value - current_value;
    const double random_number = unit_distribution_( generator_ );

    if ( random_number < std::exp( -delta_energy / cooling_function_->temperature( time_ ) ) )
    {
      current_candidate_ = *best_candidate;
    }
  }
  time_++;
  return {};
}


OptimizerMethod::ConvergenceStatus SimulatedAnnealingMethod::convergenceStatus() const
{
  if ( cooling_function_->temperature( time_ ) < hyperparameters_.minimal_temperature )
  {
    return ConvergenceStatus::Converged;
  }
  return ConvergenceStatus::InProgress;
}


void SimulatedAnnealingMethod::reset()
{
  OptimizerMethod::reset();
  neighborhood_->reset();
  generator_.seed( random_seed_ );
  unit_distribution_.reset();
  time_ = 1;
  has_current_candidate_ = false;
}
