#include "Optimizer.hpp"
#include <eigen3/Eigen/Core>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include "ConfigurationEvaluationError.hpp"
#include "OptimizationTerminationReason.hpp"
#include "OptimizerResult.hpp"

namespace
{
  void updateOptimizationCounts( OptimizerResult &result,
                                 const std::vector< CandidateEvaluation > &candidates_evaluations,
                                 bool iteration_completed )
  {
    result.number_of_batches++;
    if ( iteration_completed )
    {
      result.number_of_iterations++;
    }
    // number_of_evaluations and MaximumEvaluations count attempted candidates, including bounds rejections.
    result.number_of_evaluations += candidates_evaluations.size();
  }

  bool sameParameters( const CandidateEvaluation &left, const CandidateEvaluation &right )
  {
    return left.parameters.size() == right.parameters.size() &&
      ( left.parameters.array() == right.parameters.array() ).all();
  }

  void updateBestCandidateHistory( OptimizerResult &result, const CandidateEvaluation &candidate )
  {
    if ( result.best_parameters_history.empty() || !sameParameters( result.best_parameters_history.back(), candidate ) )
    {
      result.best_parameters_history.push_back( candidate );
    }
    else
    {
      // A stochastic method can refine the aggregate estimate at identical
      // incumbent parameters.
      result.best_parameters_history.back() = candidate;
    }

    result.best_candidate = candidate;
  }
} // namespace


Optimizer::Optimizer( std::unique_ptr< BlackBoxProblem > problem, std::unique_ptr< OptimizerMethod > method,
                      OptimizerConfiguration configuration,
                      std::unique_ptr< InitialParametersStrategy > initial_parameters_strategy ) :
    problem_( std::move( problem ) ), initial_parameters_strategy_( std::move( initial_parameters_strategy ) ),
    method_( std::move( method ) ), configuration_( std::move( configuration ) )
{
  if ( !problem_ )
  {
    throw std::invalid_argument( "Optimizer: problem cannot be null" );
  }
  if ( !method_ )
  {
    throw std::invalid_argument( "Optimizer: optimization method cannot be null" );
  }

  if ( !initial_parameters_strategy_ )
  {
    RandomInitializationConfiguration initialization_configuration;
    initialization_configuration.random_lower_bound = problem_->lowerBound();
    initialization_configuration.random_upper_bound = problem_->upperBound();
    initial_parameters_strategy_ = std::make_unique< RandomInitialization >(
      problem_->parametersCount(), std::move( initialization_configuration ) );
  }
  method_->setInitialParametersStrategy( initial_parameters_strategy_.get() );
}


CandidateEvaluation Optimizer::evaluateCandidate( BlackBoxProblem &problem, const Eigen::VectorXd &candidate )
{
  if ( candidate.size() != static_cast< Eigen::Index >( problem.parametersCount() ) )
  {
    throw std::invalid_argument( "Optimizer: candidate parameter count does not match the problem" );
  }

  if ( !candidate.allFinite() || !problem.withinBounds( candidate ) )
  {
    return CandidateEvaluation{ candidate, ObjectiveEvaluation{}, CandidateEvaluationStatus::Failed };
  }

  ObjectiveEvaluation evaluation = problem.evaluate( candidate );
  if ( evaluation.sampleCount() == 0 || !evaluation.allFinite() )
  {
    return CandidateEvaluation{ candidate, std::move( evaluation ), CandidateEvaluationStatus::Failed };
  }

  if ( problem.direction() == OptimizationDirection::Maximize )
  {
    evaluation = evaluation.negated();
  }

  return CandidateEvaluation{ candidate, std::move( evaluation ), CandidateEvaluationStatus::Succeeded };
}


std::vector< CandidateEvaluation >
Optimizer::evaluateCandidates( const std::vector< Eigen::VectorXd > &candidates )
{
  std::vector< CandidateEvaluation > evaluations;
  evaluations.reserve( candidates.size() );
  for ( const Eigen::VectorXd &candidate : candidates )
  {
    evaluations.push_back( evaluateCandidate( *problem_, candidate ) );
  }
  return evaluations;
}


OptimizerResult Optimizer::optimize()
{
  initial_parameters_strategy_->reset();
  method_->reset();

  if ( configuration_.stopping.empty() &&
       method_->convergenceStatus() == OptimizerMethod::ConvergenceStatus::NoInternalCriterion )
  {
    throw std::invalid_argument( "Optimizer requires at least one stopping criterion for this method" );
  }

  const OptimizationDirection direction = problem_->direction();
  const bool maximize = direction == OptimizationDirection::Maximize;
  OptimizationStoppingCriterion stopping_criterion( configuration_.stopping, direction );

  OptimizerResult result;
  result.number_of_iterations = 0;
  result.number_of_batches = 0;
  result.number_of_evaluations = 0;

  std::vector< CandidateEvaluation > candidates_evaluations;
  while ( true )
  {
    candidates_evaluations.clear();
    std::vector< Eigen::VectorXd > candidates = method_->ask();

    if ( candidates.empty() )
    {
      throw std::runtime_error( "Optimizer: optimization method returned no candidates" );
    }

    const std::size_t requested_candidate_count = candidates.size();
    const std::size_t allowed_candidate_count = stopping_criterion.allowedEvaluationCount( requested_candidate_count );
    const bool partial_batch = allowed_candidate_count < requested_candidate_count;
    candidates.resize( allowed_candidate_count );

    candidates_evaluations = evaluateCandidates( candidates );
    if ( candidates_evaluations.size() != candidates.size() )
    {
      throw std::logic_error( "Optimizer: evaluateCandidates returned the wrong result count" );
    }
    for ( const CandidateEvaluation &candidate : candidates_evaluations )
    {
      result.execution_metrics.append( candidate.evaluation.executionMetrics() );
    }

    // A partial batch is the final iteration because it exhausts the available evaluation allowance.
    // tell() receives complete feedback batches required by concrete method state machines.
    OptimizerMethodUpdate method_update{ false };
    if ( !partial_batch )
    {
      method_update = method_->tell( candidates_evaluations );


      const CandidateEvaluation &method_best_candidate = method_->bestCandidate();
      if ( method_best_candidate.status == CandidateEvaluationStatus::Succeeded )
      {
        updateBestCandidateHistory( result, method_best_candidate );
      }
    }

    updateOptimizationCounts( result, candidates_evaluations, method_update.iteration_completed );

    const std::optional< OptimizationTerminationReason > stopping_reason = stopping_criterion.stoppingReason(
      candidates_evaluations, method_->bestCandidate(), result.number_of_iterations );

    if ( partial_batch )
    {
      result.termination_reason =
        stopping_reason.value_or( stopping_criterion.evaluationLimitReason( requested_candidate_count ) );
      break;
    }

    const bool method_has_converged = method_->convergenceStatus() == OptimizerMethod::ConvergenceStatus::Converged;

    if ( stopping_reason.has_value() )
    {
      result.termination_reason = *stopping_reason;
      break;
    }

    if ( method_has_converged )
    {
      result.termination_reason = OptimizationTerminationReason::MethodConverged;
      break;
    }
  }

  if ( result.best_parameters_history.empty() )
  {
    throw ConfigurationEvaluationError( "Optimizer: optimization finished without a successful candidate evaluation" );
  }

  if ( maximize )
  {
    result.best_candidate.evaluation = result.best_candidate.evaluation.negated();
    for ( CandidateEvaluation &candidate : result.best_parameters_history )
    {
      candidate.evaluation = candidate.evaluation.negated();
    }
  }

  return result;
}
