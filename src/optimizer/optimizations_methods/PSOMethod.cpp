#include "optimizations_methods/PSOMethod.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include "ConfigurationEvaluationError.hpp"

PSOMethod::PSOMethod( std::size_t number_of_dimensions, InitialParametersStrategy *initial_parameters_strategy ) :
    PSOMethod( PSOHyperparameters{}, number_of_dimensions, initial_parameters_strategy )
{
}


PSOMethod::PSOMethod( const PSOHyperparameters &hyperparameters, std::size_t number_of_dimensions,
                      InitialParametersStrategy *initial_parameters_strategy ) :
    OptimizerMethod( initial_parameters_strategy ), hyperparameters_( hyperparameters ),
    number_of_dimensions_( number_of_dimensions ), generator_( hyperparameters.random_seed )
{
  validateHyperparameters();
}


void PSOMethod::validateHyperparameters() const
{
  if ( hyperparameters_.population_size == 0 )
  {
    throw std::invalid_argument( "PSOMethod: population size must be greater than zero" );
  }
  if ( !std::isfinite( hyperparameters_.initial_inertia_weight ) || hyperparameters_.initial_inertia_weight < 0.0 )
  {
    throw std::invalid_argument( "PSOMethod: initial inertia weight must be finite and non-negative" );
  }
  if ( !std::isfinite( hyperparameters_.cognitive_coefficient ) || hyperparameters_.cognitive_coefficient < 0.0 )
  {
    throw std::invalid_argument( "PSOMethod: cognitive coefficient must be finite and non-negative" );
  }
  if ( !std::isfinite( hyperparameters_.social_coefficient ) || hyperparameters_.social_coefficient < 0.0 )
  {
    throw std::invalid_argument( "PSOMethod: social coefficient must be finite and non-negative" );
  }
  if ( !std::isfinite( hyperparameters_.initial_velocity_scale ) ||
       hyperparameters_.initial_velocity_scale <= 0.0 )
  {
    throw std::invalid_argument( "PSOMethod: initial velocity scale must be finite and positive" );
  }
}


void PSOMethod::initializeParticles()
{
  positions_.clear();
  personal_best_positions_.clear();
  personal_best_values_.assign( hyperparameters_.population_size, std::numeric_limits< double >::infinity() );

  positions_.reserve( hyperparameters_.population_size );
  personal_best_positions_.reserve( hyperparameters_.population_size );

  for ( std::size_t particle = 0; particle < hyperparameters_.population_size; ++particle )
  {
    Eigen::VectorXd position = initial_parameters_strategy_->generateInitialParameters();

    positions_.push_back( position );
    personal_best_positions_.push_back( std::move( position ) );
  }
}


void PSOMethod::initializeVelocities()
{
  velocities_.clear();
  velocities_.reserve( hyperparameters_.population_size );

  for ( std::size_t particle = 0; particle < hyperparameters_.population_size; ++particle )
  {
    Eigen::VectorXd velocity( static_cast< Eigen::Index >( number_of_dimensions_ ) );
    for ( Eigen::Index coordinate = 0; coordinate < velocity.size(); ++coordinate )
    {
      const double symmetric_sample = 2.0 * unit_distribution_( generator_ ) - 1.0;
      velocity( coordinate ) = symmetric_sample * hyperparameters_.initial_velocity_scale;
    }
    velocities_.push_back( std::move( velocity ) );
  }
}


void PSOMethod::updateVelocitiesAndPositions()
{
  for ( std::size_t particle = 0; particle < positions_.size(); ++particle )
  {
    Eigen::VectorXd &position = positions_.at( particle );
    Eigen::VectorXd &velocity = velocities_.at( particle );
    const Eigen::VectorXd &personal_best = personal_best_positions_.at( particle );

    for ( Eigen::Index coordinate = 0; coordinate < position.size(); ++coordinate )
    {
      const double personal_random = unit_distribution_( generator_ );
      const double global_random = unit_distribution_( generator_ );

      velocity( coordinate ) = hyperparameters_.initial_inertia_weight * velocity( coordinate ) +
        hyperparameters_.cognitive_coefficient * personal_random *
          ( personal_best( coordinate ) - position( coordinate ) ) +
        hyperparameters_.social_coefficient * global_random *
          ( global_best_position_( coordinate ) - position( coordinate ) );
    }

    position += velocity;
  }
}


void PSOMethod::updateBestPositions( const std::vector< CandidateEvaluation > &evaluations )
{
  if ( evaluations.size() != hyperparameters_.population_size )
  {
    throw std::invalid_argument( "PSOMethod: evaluation count does not match population size" );
  }

  for ( std::size_t particle = 0; particle < evaluations.size(); ++particle )
  {
    const CandidateEvaluation &candidate = evaluations.at( particle );
    if ( candidate.status != CandidateEvaluationStatus::Succeeded )
    {
      continue;
    }

    const double value = candidate.meanValue();
    if ( value < personal_best_values_.at( particle ) )
    {
      personal_best_values_.at( particle ) = value;
      personal_best_positions_.at( particle ) = candidate.parameters;
    }

    if ( !has_global_best_ || value < global_best_value_ )
    {
      global_best_value_ = value;
      global_best_position_ = candidate.parameters;
      has_global_best_ = true;
    }
  }
}


std::vector< Eigen::VectorXd > PSOMethod::ask()
{
  switch ( phase_ )
  {
    case Phase::Initialization:
      initializeParticles();
      phase_ = Phase::WaitingInitialization;
      return positions_;

    case Phase::Optimization:
      updateVelocitiesAndPositions();
      phase_ = Phase::WaitingOptimization;
      return positions_;

    default:
      throw std::logic_error( "PSOMethod: ask called while waiting for evaluations" );
  }
}


OptimizerMethodUpdate PSOMethod::tellImpl( const std::vector< CandidateEvaluation > &evaluations )
{
  switch ( phase_ )
  {
    case Phase::WaitingInitialization:
      updateBestPositions( evaluations );
      if ( !has_global_best_ )
      {
        throw ConfigurationEvaluationError(
          "PSOMethod: initialization produced no successful particle evaluation" );
      }
      initializeVelocities();
      phase_ = Phase::Optimization;
      return OptimizerMethodUpdate{ false };

    case Phase::WaitingOptimization:
      updateBestPositions( evaluations );
      phase_ = Phase::Optimization;
      return {};

    default:
      throw std::logic_error( "PSOMethod: tell called without pending evaluations" );
  }
}


void PSOMethod::reset()
{
  OptimizerMethod::reset();

  positions_.clear();
  velocities_.clear();
  personal_best_positions_.clear();
  personal_best_values_.clear();
  global_best_position_.resize( 0 );
  global_best_value_ = 0.0;
  has_global_best_ = false;

  generator_.seed( hyperparameters_.random_seed );
  unit_distribution_.reset();
  phase_ = Phase::Initialization;
}
