#include "ModelSelectionStoppingCriterion.hpp"
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>
#include "ConfigurationSelector.hpp"
#include "ModelSelectionStoppingConfiguration.hpp"
#include "ModelSelectionStoppingCriterionFactory.hpp"
#include "ModelSelectionStoppingStrategy.hpp"
#include "OptimizationTerminationReason.hpp"

ModelSelectionStoppingCriterion::ModelSelectionStoppingCriterion( ModelSelectionStoppingConfiguration configuration )
{
  addCriterion( configuration );
}

ModelSelectionStoppingCriterion::ModelSelectionStoppingCriterion(
  const std::vector< ModelSelectionStoppingConfiguration > &configurations )
{
  for ( const ModelSelectionStoppingConfiguration &configuration : configurations )
  {
    addCriterion( configuration );
  }
}

std::optional< OptimizationTerminationReason >
ModelSelectionStoppingCriterion::stoppingReason( const ConfigurationFeedback &current_feedback )
{
  for ( const std::size_t criterion_index : criterion_order_ )
  {
    const std::unique_ptr< ModelSelectionStoppingStrategy > &criterion = criteria_.at( criterion_index );
    if ( criterion->shouldStop( current_feedback ) )
    {
      return criterion->terminationReason();
    }
  }

  return std::nullopt;
}


void ModelSelectionStoppingCriterion::addCriterion( const ModelSelectionStoppingConfiguration &configuration )
{
  std::unique_ptr< ModelSelectionStoppingStrategy > &criterion = criteria_.at( configuration.index() );

  if ( criterion )
  {
    throw std::invalid_argument( "ModelSelectionStoppingCriterion: duplicate stopping criterion type" );
  }

  criterion = ModelSelectionStoppingCriterionFactory::create( configuration );
  criterion_order_.push_back( configuration.index() );
}
