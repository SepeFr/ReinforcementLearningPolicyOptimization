#include "optimizations_methods/SPSAMethod.hpp"
#include <cmath>
#include <stdexcept>

SPSAMethod::SPSAMethod( std::size_t number_of_dimensions, InitialParametersStrategy *initial_parameters_strategy ) :
    SPSAMethod( SPSAHyperparameters{}, number_of_dimensions, initial_parameters_strategy )
{
}


SPSAMethod::SPSAMethod( const SPSAHyperparameters &hyperparameters, std::size_t number_of_dimensions,
                        InitialParametersStrategy *initial_parameters_strategy ) :
    OptimizerMethod( initial_parameters_strategy ), hyperparameters_( hyperparameters ),
    number_of_dimensions_( number_of_dimensions ), generator_( hyperparameters.random_seed )
{
  if ( !std::isfinite( hyperparameters_.perturbation_magnitude ) || hyperparameters_.perturbation_magnitude <= 0.0 )
  {
    throw std::invalid_argument( "SPSAMethod: perturbation magnitude must be finite and positive" );
  }
  if ( !std::isfinite( hyperparameters_.step_size ) || hyperparameters_.step_size <= 0.0 )
  {
    throw std::invalid_argument( "SPSAMethod: step size must be finite and positive" );
  }
}


void SPSAMethod::initializeParameters()
{
  current_parameters_ = initial_parameters_strategy_->generateInitialParameters();
  if ( current_parameters_.size() != static_cast< Eigen::Index >( number_of_dimensions_ ) )
  {
    throw std::invalid_argument( "SPSAMethod: initial parameter dimension does not match the problem" );
  }

  perturbation_vector_.resize( current_parameters_.size() );
  initialized_ = true;
}


void SPSAMethod::generatePerturbation()
{
  for ( Eigen::Index index = 0; index < perturbation_vector_.size(); ++index )
  {
    perturbation_vector_( index ) = unit_distribution_( generator_ ) < 0.5 ? -1.0 : 1.0;
  }
}


std::vector< Eigen::VectorXd > SPSAMethod::ask()
{
  if ( waiting_for_evaluations_ )
  {
    throw std::logic_error( "SPSAMethod: ask called while waiting for evaluations" );
  }
  if ( !initialized_ )
  {
    initializeParameters();
  }

  generatePerturbation();
  waiting_for_evaluations_ = true;

  return { current_parameters_ + hyperparameters_.perturbation_magnitude * perturbation_vector_,
           current_parameters_ - hyperparameters_.perturbation_magnitude * perturbation_vector_ };
}


OptimizerMethodUpdate SPSAMethod::tellImpl( const std::vector< CandidateEvaluation > &evaluations )
{
  if ( !waiting_for_evaluations_ )
  {
    throw std::logic_error( "SPSAMethod: tell called without pending evaluations" );
  }
  if ( evaluations.size() != 2 )
  {
    throw std::invalid_argument( "SPSAMethod: tell requires exactly two evaluations" );
  }

  if ( evaluations.front().status == CandidateEvaluationStatus::Succeeded &&
       evaluations.back().status == CandidateEvaluationStatus::Succeeded )
  {
    const Eigen::VectorXd reciprocal_perturbation = perturbation_vector_.cwiseInverse();
    const double objective_difference = evaluations.front().meanValue() - evaluations.back().meanValue();
    const Eigen::VectorXd gradient =
      objective_difference / ( 2.0 * hyperparameters_.perturbation_magnitude ) * reciprocal_perturbation;

    current_parameters_ -= hyperparameters_.step_size * gradient;
  }

  waiting_for_evaluations_ = false;
  return {};
}


void SPSAMethod::reset()
{
  OptimizerMethod::reset();

  current_parameters_.resize( 0 );
  perturbation_vector_.resize( 0 );
  generator_.seed( hyperparameters_.random_seed );
  unit_distribution_.reset();
  initialized_ = false;
  waiting_for_evaluations_ = false;
}
