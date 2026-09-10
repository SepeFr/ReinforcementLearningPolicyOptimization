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
    current_noise_scale_( hyperparameters_.noise_scale ), generator_( hyperparameters_.random_seed )
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
  if ( hyperparameters_.adapt_noise_scale &&
       ( !std::isfinite( hyperparameters_.target_success_rate ) || hyperparameters_.target_success_rate <= 0.0 ||
         hyperparameters_.target_success_rate >= 1.0 ||
         !std::isfinite( hyperparameters_.noise_scale_increase_factor ) ||
         hyperparameters_.noise_scale_increase_factor <= 1.0 ||
         !std::isfinite( hyperparameters_.noise_scale_decrease_factor ) ||
         hyperparameters_.noise_scale_decrease_factor <= 0.0 || hyperparameters_.noise_scale_decrease_factor >= 1.0 ||
         !std::isfinite( hyperparameters_.minimum_noise_scale ) || hyperparameters_.minimum_noise_scale <= 0.0 ||
         !std::isfinite( hyperparameters_.maximum_noise_scale ) ||
         hyperparameters_.maximum_noise_scale < hyperparameters_.minimum_noise_scale ||
         hyperparameters_.noise_scale < hyperparameters_.minimum_noise_scale ||
         hyperparameters_.noise_scale > hyperparameters_.maximum_noise_scale ) )
  {
    throw std::invalid_argument( "OpenAIESMethod: invalid adaptive noise-scale configuration" );
  }
  if ( !std::isfinite( hyperparameters_.beta_1 ) || hyperparameters_.beta_1 < 0.0 || hyperparameters_.beta_1 >= 1.0 ||
       !std::isfinite( hyperparameters_.beta_2 ) || hyperparameters_.beta_2 < 0.0 || hyperparameters_.beta_2 >= 1.0 )
  {
    throw std::invalid_argument( "OpenAIESMethod: Adam betas must be finite and in [0, 1)" );
  }
  if ( !std::isfinite( hyperparameters_.epsilon ) || hyperparameters_.epsilon <= 0.0 )
  {
    throw std::invalid_argument( "OpenAIESMethod: Adam epsilon must be finite and positive" );
  }
  if ( !std::isfinite( hyperparameters_.weight_decay ) || hyperparameters_.weight_decay < 0.0 )
  {
    throw std::invalid_argument( "OpenAIESMethod: AdamW weight decay must be finite and non-negative" );
  }
}


void OpenAIESMethod::initializeParameters()
{
  current_parameters_ = initial_parameters_strategy_->generateInitialParameters();
  if ( current_parameters_.size() == 0 )
  {
    throw std::invalid_argument( "OpenAIESMethod: initial parameters cannot be empty" );
  }
  if ( hyperparameters_.frozen_prefix_size > static_cast< std::size_t >( current_parameters_.size() ) )
  {
    throw std::invalid_argument( "OpenAIESMethod: frozen prefix exceeds parameter count" );
  }

  if ( pending_initial_center_state_.has_value() )
  {
    validateCenterState( *pending_initial_center_state_, current_parameters_.size() );
    current_parameters_ = pending_initial_center_state_->parameters;
    adam_first_moment_ = pending_initial_center_state_->adam_first_moment;
    adam_second_moment_ = pending_initial_center_state_->adam_second_moment;
    adam_step_ = pending_initial_center_state_->adam_step;
    current_noise_scale_ = pending_initial_center_state_->noise_scale;
    pending_initial_center_state_.reset();
  }

  initialized_ = true;
}


Eigen::VectorXd OpenAIESMethod::sampleStandardNoise()
{
  Eigen::VectorXd noise( current_parameters_.size() );
  noise.setZero();
  for ( Eigen::Index index = static_cast< Eigen::Index >( hyperparameters_.frozen_prefix_size ); index < noise.size();
        ++index )
  {
    noise( index ) = standard_normal_distribution_( generator_ );
  }
  return noise;
}


const Eigen::VectorXd &OpenAIESMethod::currentCenter() const
{
  if ( !initialized_ || current_parameters_.size() == 0 )
  {
    throw std::logic_error( "OpenAIESMethod: current center is unavailable before initialization" );
  }

  return current_parameters_;
}


OpenAIESMethod::CenterState OpenAIESMethod::centerState() const
{
  if ( !initialized_ || current_parameters_.size() == 0 )
  {
    throw std::logic_error( "OpenAIESMethod: center state is unavailable before initialization" );
  }

  return CenterState{ current_parameters_, adam_first_moment_, adam_second_moment_, adam_step_, current_noise_scale_ };
}


void OpenAIESMethod::validateCenterState( const CenterState &state, Eigen::Index expected_parameter_count ) const
{
  const bool has_moments = state.adam_first_moment.size() != 0;
  if ( state.parameters.size() != expected_parameter_count ||
       state.adam_first_moment.size() != state.adam_second_moment.size() ||
       ( has_moments && state.adam_first_moment.size() != state.parameters.size() ) ||
       has_moments != ( state.adam_step != 0 ) || !state.parameters.allFinite() ||
       ( has_moments && ( !state.adam_first_moment.allFinite() || !state.adam_second_moment.allFinite() ) ) ||
       !std::isfinite( state.noise_scale ) || state.noise_scale <= 0.0 ||
       ( hyperparameters_.adapt_noise_scale &&
         ( state.noise_scale < hyperparameters_.minimum_noise_scale ||
           state.noise_scale > hyperparameters_.maximum_noise_scale ) ) )
  {
    throw std::invalid_argument( "OpenAIESMethod: restored center state is incompatible" );
  }
}


void OpenAIESMethod::setInitialCenterState( const CenterState &state )
{
  if ( initialized_ || waiting_for_evaluations_ )
  {
    throw std::logic_error( "OpenAIESMethod: initial center state must be supplied before optimization" );
  }
  pending_initial_center_state_ = state;
}


void OpenAIESMethod::restoreCenterState( const CenterState &state )
{
  if ( !initialized_ || waiting_for_evaluations_ )
  {
    throw std::logic_error( "OpenAIESMethod: center state cannot be restored while feedback is pending" );
  }
  validateCenterState( state, current_parameters_.size() );

  current_parameters_ = state.parameters;
  adam_first_moment_ = state.adam_first_moment;
  adam_second_moment_ = state.adam_second_moment;
  adam_step_ = state.adam_step;
  current_noise_scale_ = state.noise_scale;
}


std::vector< double > OpenAIESMethod::fitnessValues( const std::vector< CandidateEvaluation > &evaluations ) const
{
  if ( hyperparameters_.use_rank_fitness )
  {
    return centeredRankFitness( evaluations );
  }

  std::vector< double > fitness( evaluations.size() - 1, 0.0 );
  for ( std::size_t index = 0; index < fitness.size(); ++index )
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
  successful_indices.reserve( evaluations.size() - 1 );

  for ( std::size_t index = 0; index + 1 < evaluations.size(); ++index )
  {
    if ( evaluations.at( index ).status == CandidateEvaluationStatus::Succeeded )
    {
      successful_indices.push_back( index );
    }
  }

  std::sort( successful_indices.begin(), successful_indices.end(), [&]( std::size_t left, std::size_t right )
             { return evaluations.at( left ).meanValue() < evaluations.at( right ).meanValue(); } );


  std::vector< double > fitness( evaluations.size() - 1, 0.0 );
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
  candidates.reserve( hyperparameters_.population_size + 1 );

  for ( std::size_t direction = 0; direction < direction_count; ++direction )
  {
    Eigen::VectorXd noise = sampleStandardNoise();
    sampled_noises_.push_back( noise );
    candidates.push_back( current_parameters_ + current_noise_scale_ * noise );
    candidates.push_back( current_parameters_ - current_noise_scale_ * noise );
  }
  candidates.push_back( current_parameters_ );

  waiting_for_evaluations_ = true;
  return candidates;
}


void OpenAIESMethod::updateParameters( const Eigen::VectorXd &search_direction )
{
  switch ( hyperparameters_.update_method )
  {
    case OpenAIESHyperparameters::UpdateMethod::SGD:
      updateWithSGD( search_direction );
      break;
    case OpenAIESHyperparameters::UpdateMethod::Adam:
      updateWithAdam( search_direction );
      break;
    case OpenAIESHyperparameters::UpdateMethod::AdamW:
      updateWithAdamW( search_direction );
      break;
  }
}


void OpenAIESMethod::updateWithSGD( const Eigen::VectorXd &search_direction )
{
  current_parameters_ += hyperparameters_.learning_rate * search_direction;
}


void OpenAIESMethod::updateWithAdam( const Eigen::VectorXd &search_direction )
{
  current_parameters_ += hyperparameters_.learning_rate * adamSearchStep( search_direction );
}


void OpenAIESMethod::updateWithAdamW( const Eigen::VectorXd &search_direction )
{
  const Eigen::VectorXd parameters_before_update = current_parameters_;
  const Eigen::VectorXd adaptive_step = adamSearchStep( search_direction );
  const Eigen::Index first_active_parameter = static_cast< Eigen::Index >( hyperparameters_.frozen_prefix_size );
  const Eigen::Index active_parameter_count = current_parameters_.size() - first_active_parameter;

  // AdamW decouples regularization from the noisy ES search direction. Do not
  // decay a configured frozen prefix: it is intentionally held constant.
  if ( active_parameter_count > 0 )
  {
    current_parameters_.tail( active_parameter_count ) += hyperparameters_.learning_rate *
      ( adaptive_step.tail( active_parameter_count ) -
        hyperparameters_.weight_decay * parameters_before_update.tail( active_parameter_count ) );
  }
}


Eigen::VectorXd OpenAIESMethod::adamSearchStep( const Eigen::VectorXd &search_direction )
{
  if ( adam_first_moment_.size() == 0 )
  {
    adam_first_moment_ = Eigen::VectorXd::Zero( current_parameters_.size() );
    adam_second_moment_ = Eigen::VectorXd::Zero( current_parameters_.size() );
  }

  adam_first_moment_ =
    hyperparameters_.beta_1 * adam_first_moment_ + ( 1.0 - hyperparameters_.beta_1 ) * search_direction;
  adam_second_moment_ = hyperparameters_.beta_2 * adam_second_moment_ +
    ( 1.0 - hyperparameters_.beta_2 ) * search_direction.array().square().matrix();
  ++adam_step_;

  const double first_moment_correction = 1.0 - std::pow( hyperparameters_.beta_1, adam_step_ );
  const double second_moment_correction = 1.0 - std::pow( hyperparameters_.beta_2, adam_step_ );
  const Eigen::VectorXd corrected_first_moment = adam_first_moment_ / first_moment_correction;
  const Eigen::VectorXd corrected_second_moment = adam_second_moment_ / second_moment_correction;
  return ( corrected_first_moment.array() / ( corrected_second_moment.array().sqrt() + hyperparameters_.epsilon ) )
    .matrix();
}


void OpenAIESMethod::adaptNoiseScale( const std::vector< CandidateEvaluation > &evaluations )
{
  if ( !hyperparameters_.adapt_noise_scale || evaluations.size() < 2 )
  {
    return;
  }

  const CandidateEvaluation &parent = evaluations.back();
  const std::size_t trials = evaluations.size() - 1;
  std::size_t successes = 0;
  for ( std::size_t index = 0; index < trials; ++index )
  {
    const CandidateEvaluation &offspring = evaluations.at( index );
    if ( offspring.status != CandidateEvaluationStatus::Succeeded )
    {
      continue;
    }

    bool better = parent.status != CandidateEvaluationStatus::Succeeded;
    if ( !better )
    {
      const std::size_t offspring_failures = offspring.evaluation.failedSampleCount();
      const std::size_t parent_failures = parent.evaluation.failedSampleCount();
      better = offspring_failures < parent_failures ||
        ( offspring_failures == parent_failures && offspring.meanValue() < parent.meanValue() );
    }
    successes += static_cast< std::size_t >( better );
  }

  const double success_rate = static_cast< double >( successes ) / static_cast< double >( trials );
  last_mutation_success_rate_ = success_rate;
  if ( success_rate > hyperparameters_.target_success_rate )
  {
    current_noise_scale_ *= hyperparameters_.noise_scale_increase_factor;
  }
  else if ( success_rate < hyperparameters_.target_success_rate )
  {
    current_noise_scale_ *= hyperparameters_.noise_scale_decrease_factor;
  }
  current_noise_scale_ =
    std::clamp( current_noise_scale_, hyperparameters_.minimum_noise_scale, hyperparameters_.maximum_noise_scale );
}


OptimizerMethodUpdate OpenAIESMethod::tellImpl( const std::vector< CandidateEvaluation > &evaluations )
{
  if ( !waiting_for_evaluations_ )
  {
    throw std::logic_error( "OpenAIESMethod: tell called without pending evaluations" );
  }
  if ( evaluations.size() != 2 * sampled_noises_.size() + 1 )
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

  search_direction /= static_cast< double >( hyperparameters_.population_size ) * current_noise_scale_;

  // The score-function gradient is normalized by both population size and perturbation scale.
  updateParameters( search_direction );
  adaptNoiseScale( evaluations );
  waiting_for_evaluations_ = false;
  return {};
}


void OpenAIESMethod::updateBestCandidate( const std::vector< CandidateEvaluation > &evaluations )
{
  const CandidateEvaluation &center = evaluations.back();
  if ( center.status == CandidateEvaluationStatus::Succeeded &&
       ( best_candidate_evaluation_.status != CandidateEvaluationStatus::Succeeded ||
         center.meanValue() < best_candidate_evaluation_.meanValue() ) )
  {
    best_candidate_evaluation_ = center;
  }
}


void OpenAIESMethod::reset()
{
  OptimizerMethod::reset();
  generator_.seed( hyperparameters_.random_seed );
  standard_normal_distribution_.reset();
  current_parameters_.resize( 0 );
  adam_first_moment_.resize( 0 );
  adam_second_moment_.resize( 0 );
  adam_step_ = 0;
  current_noise_scale_ = hyperparameters_.noise_scale;
  last_mutation_success_rate_.reset();
  sampled_noises_.clear();
  initialized_ = false;
  waiting_for_evaluations_ = false;
}
