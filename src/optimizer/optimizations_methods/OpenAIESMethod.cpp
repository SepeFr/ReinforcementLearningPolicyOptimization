#include "optimizations_methods/OpenAIESMethod.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <stdexcept>
#include <vector>
#include "CandidateEvaluation.hpp"


OpenAIESMethod::OpenAIESMethod( InitialParametersStrategy *initial_parameters_strategy ) :
    OpenAIESMethod( OpenAIESHyperparameters{}, initial_parameters_strategy )
{
}


OpenAIESMethod::OpenAIESMethod( const OpenAIESHyperparameters &hyperparameters,
                                InitialParametersStrategy *initial_parameters_strategy ) :
    OptimizerMethod( initial_parameters_strategy ), hyperparameters_( hyperparameters ),
    generator_( hyperparameters_.random_seed )
{
  if ( hyperparameters_.population_size < 2 || hyperparameters_.population_size % 2 != 0 )
  {
    throw std::invalid_argument( "OpenAIESMethod: population size must be even and at least two" );
  }
  if ( !std::isfinite( hyperparameters_.learning_rate ) || hyperparameters_.learning_rate <= 0.0 )
  {
    throw std::invalid_argument( "OpenAIESMethod: learning rate must be finite and positive" );
  }
  if ( !std::isfinite( hyperparameters_.noise_scale ) || hyperparameters_.noise_scale <= 0.0 )
  {
    throw std::invalid_argument( "OpenAIESMethod: noise scale must be finite and positive" );
  }
}


void OpenAIESMethod::initializeParameters()
{
  current_parameters_ = initial_parameters_strategy_->generateInitialParameters();
  if ( current_parameters_.size() == 0 )
  {
    throw std::invalid_argument( "OpenAIESMethod: initial parameters cannot be empty" );
  }

  initialized_ = true;
}


Eigen::VectorXd OpenAIESMethod::sampleStandardNoise()
{
  Eigen::VectorXd noise( current_parameters_.size() );
  for ( Eigen::Index index = 0; index < noise.size(); ++index )
  {
    noise( index ) = standard_normal_distribution_( generator_ );
  }
  return noise;
}


std::vector< double > OpenAIESMethod::fitnessValues( const std::vector< CandidateEvaluation > &evaluations ) const
{
  if ( hyperparameters_.use_rank_fitness )
  {
    return centeredRankFitness( evaluations );
  }

  std::vector< double > fitness( evaluations.size(), 0.0 );
  for ( std::size_t index = 0; index < evaluations.size(); ++index )
  {
    if ( evaluations.at( index ).status == CandidateEvaluationStatus::Succeeded )
    {
      // The library minimizes internally, so -f is the utility maximized by ES.
      fitness.at( index ) = -evaluations.at( index ).meanValue();
    }
  }
  return fitness;
}


std::vector< double > OpenAIESMethod::centeredRankFitness( const std::vector< CandidateEvaluation > &evaluations ) const
{
  std::vector< std::size_t > successful_indices;
  successful_indices.reserve( evaluations.size() );

  for ( std::size_t index = 0; index < evaluations.size(); ++index )
  {
    if ( evaluations.at( index ).status == CandidateEvaluationStatus::Succeeded )
    {
      successful_indices.push_back( index );
    }
  }

  std::sort( successful_indices.begin(), successful_indices.end(), [&]( std::size_t left, std::size_t right )
             { return evaluations.at( left ).meanValue() < evaluations.at( right ).meanValue(); } );

  std::vector< double > fitness( evaluations.size(), 0.0 );
  if ( successful_indices.size() < 2 )
  {
    return fitness;
  }

  const double rank_scale = static_cast< double >( successful_indices.size() - 1 );

  // OpenAI centered ranks, reversed because the library minimizes internally.
  for ( std::size_t rank = 0; rank < successful_indices.size(); ++rank )
  {
    fitness.at( successful_indices.at( rank ) ) = 0.5 - static_cast< double >( rank ) / rank_scale;
  }

  return fitness;
}


std::vector< Eigen::VectorXd > OpenAIESMethod::ask()
{
  if ( waiting_for_evaluations_ )
  {
    throw std::logic_error( "OpenAIESMethod: ask called while waiting for evaluations" );
  }
  if ( !initialized_ )
  {
    initializeParameters();
  }

  sampled_noises_.clear();
  const std::size_t direction_count = hyperparameters_.population_size / 2;
  sampled_noises_.reserve( direction_count );

  std::vector< Eigen::VectorXd > candidates;
  candidates.reserve( hyperparameters_.population_size );

  for ( std::size_t direction = 0; direction < direction_count; ++direction )
  {
    Eigen::VectorXd noise = sampleStandardNoise();
    sampled_noises_.push_back( noise );
    candidates.push_back( current_parameters_ + hyperparameters_.noise_scale * noise );
    candidates.push_back( current_parameters_ - hyperparameters_.noise_scale * noise );
  }

  waiting_for_evaluations_ = true;
  return candidates;
}


OptimizerMethodUpdate OpenAIESMethod::tellImpl( const std::vector< CandidateEvaluation > &evaluations )
{
  if ( !waiting_for_evaluations_ )
  {
    throw std::logic_error( "OpenAIESMethod: tell called without pending evaluations" );
  }
  if ( evaluations.size() != 2 * sampled_noises_.size() )
  {
    throw std::invalid_argument( "OpenAIESMethod: evaluation count does not match population size" );
  }

  const std::vector< double > fitness = fitnessValues( evaluations );
  Eigen::VectorXd search_direction = Eigen::VectorXd::Zero( current_parameters_.size() );

  for ( std::size_t direction = 0; direction < sampled_noises_.size(); ++direction )
  {
    const std::size_t positive_index = 2 * direction;
    const std::size_t negative_index = positive_index + 1;

    if ( evaluations.at( positive_index ).status != CandidateEvaluationStatus::Succeeded ||
         evaluations.at( negative_index ).status != CandidateEvaluationStatus::Succeeded )
    {
      continue;
    }

    search_direction +=
      ( fitness.at( positive_index ) - fitness.at( negative_index ) ) * sampled_noises_.at( direction );
  }

  search_direction /=
    static_cast< double >( hyperparameters_.population_size ) * hyperparameters_.noise_scale;

  // The score-function gradient is normalized by both population size and perturbation scale.
  current_parameters_ += hyperparameters_.learning_rate * search_direction;
  waiting_for_evaluations_ = false;
  return {};
}


void OpenAIESMethod::reset()
{
  OptimizerMethod::reset();
  generator_.seed( hyperparameters_.random_seed );
  standard_normal_distribution_.reset();
  current_parameters_.resize( 0 );
  sampled_noises_.clear();
  initialized_ = false;
  waiting_for_evaluations_ = false;
}
