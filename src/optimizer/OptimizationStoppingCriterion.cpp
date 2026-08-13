#include "OptimizationStoppingCriterion.hpp"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>
#include "CandidateEvaluation.hpp"
#include "OptimizationStoppingStrategy.hpp"
#include "OptimizationTerminationReason.hpp"
#include "StoppingCriterionFactory.hpp"

OptimizationStoppingCriterion::OptimizationStoppingCriterion( StoppingConfiguration configuration,
                                                                OptimizationDirection direction )
{
  addCriterion( configuration, direction );
}


OptimizationStoppingCriterion::OptimizationStoppingCriterion(
  const std::vector< StoppingConfiguration > &configurations, OptimizationDirection direction )
{
  for ( const StoppingConfiguration &configuration : configurations )
  {
    addCriterion( configuration, direction );
  }
}


std::size_t OptimizationStoppingCriterion::allowedEvaluationCount( std::size_t requested_count ) const
{
  std::size_t allowed_count = requested_count;

  for ( const std::size_t criterion_index : criterion_order_ )
  {
    const std::unique_ptr< OptimizationStoppingStrategy > &criterion = criteria_.at( criterion_index );
    allowed_count = std::min( allowed_count, criterion->allowedEvaluationCount( requested_count ) );
  }

  return allowed_count;
}


OptimizationTerminationReason
OptimizationStoppingCriterion::evaluationLimitReason( std::size_t requested_count ) const
{
  for ( const std::size_t criterion_index : criterion_order_ )
  {
    const std::unique_ptr< OptimizationStoppingStrategy > &criterion = criteria_.at( criterion_index );
    if ( criterion->allowedEvaluationCount( requested_count ) < requested_count )
    {
      return criterion->terminationReason();
    }
  }

  return OptimizationTerminationReason::None;
}


std::optional< OptimizationTerminationReason >
OptimizationStoppingCriterion::stoppingReason(
  const std::vector< CandidateEvaluation > &current_evaluations,
  const CandidateEvaluation &best_candidate,
  std::size_t method_iterations )
{
  for ( const std::size_t criterion_index : criterion_order_ )
  {
    const std::unique_ptr< OptimizationStoppingStrategy > &criterion = criteria_.at( criterion_index );
    if ( criterion->shouldStop( current_evaluations, best_candidate, method_iterations ) )
    {
      return criterion->terminationReason();
    }
  }

  return std::nullopt;
}


void OptimizationStoppingCriterion::addCriterion( const StoppingConfiguration &configuration,
                                                   OptimizationDirection direction )
{
  std::unique_ptr< OptimizationStoppingStrategy > &criterion = criteria_.at( configuration.index() );

  if ( criterion )
  {
    throw std::invalid_argument( "OptimizationStoppingCriterion: duplicate stopping criterion type" );
  }

  criterion = StoppingCriterionFactory::create( configuration, direction );
  criterion_order_.push_back( configuration.index() );
}
