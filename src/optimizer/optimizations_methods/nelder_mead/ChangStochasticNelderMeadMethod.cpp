#include "optimizations_methods/nelder_mead/ChangStochasticNelderMeadMethod.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "ConfigurationEvaluationError.hpp"

ChangStochasticNelderMeadMethod::ChangStochasticNelderMeadMethod(
  std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy ) :
    ChangStochasticNelderMeadMethod( ChangStochasticNelderMeadHyperparameters{}, number_of_parameters,
                                     initial_parameters_strategy )
{
}

ChangStochasticNelderMeadMethod::ChangStochasticNelderMeadMethod(
  const ChangStochasticNelderMeadHyperparameters &hyperparameters, std::size_t number_of_parameters,
  InitialParametersStrategy *initial_parameters_strategy ) :
    NelderMeadMethod( hyperparameters, number_of_parameters, initial_parameters_strategy ),
    hyperparameters_( hyperparameters ), generator_( hyperparameters.random_seed )
{
  if ( hyperparameters_.minimum_sample_size == 0 )
  {
    throw std::invalid_argument( "ChangStochasticNelderMeadMethod: minimum sample size must be greater than zero" );
  }
  if ( !std::isfinite( hyperparameters_.global_search_probability ) ||
       hyperparameters_.global_search_probability <= 0.0 || hyperparameters_.global_search_probability >= 1.0 )
  {
    throw std::invalid_argument( "ChangStochasticNelderMeadMethod: global search probability must be in (0, 1)" );
  }
}

std::size_t ChangStochasticNelderMeadMethod::targetSampleCount() const
{
  // Chang's basic schedule is N_k = ceil(sqrt(k)); k + 1 keeps the
  // initialization well-defined when the internal iteration starts at zero.
  const double scheduled_count = std::ceil( std::sqrt( static_cast< double >( iterationCount() + 1 ) ) );
  return std::max( hyperparameters_.minimum_sample_size, static_cast< std::size_t >( scheduled_count ) );
}

std::vector< Eigen::VectorXd > ChangStochasticNelderMeadMethod::requestSimplexRefresh()
{
  const std::size_t target_sample_count = targetSampleCount();
  std::vector< Eigen::VectorXd > requests;
  pending_candidates_.clear();
  pending_candidates_.reserve( simplex().size() );

  for ( const CandidateEvaluation &vertex : simplex() )
  {
    // Existing vertices retain their observations and request only the
    // additional samples needed to reach the current N_k.
    const std::size_t current_sample_count =
      vertex.status == CandidateEvaluationStatus::Succeeded ? vertex.evaluation.sampleCount() : 0;
    const std::size_t missing_sample_count =
      target_sample_count > current_sample_count ? target_sample_count - current_sample_count : 0;

    pending_candidates_.push_back( PendingCandidate{ vertex, missing_sample_count } );
    for ( std::size_t sample = 0; sample < missing_sample_count; ++sample )
    {
      requests.push_back( vertex.parameters );
    }
  }

  return requests;
}

std::vector< Eigen::VectorXd >
ChangStochasticNelderMeadMethod::requestNewCandidates( const std::vector< Eigen::VectorXd > &logical_candidates )
{
  const std::size_t target_sample_count = targetSampleCount();
  std::vector< Eigen::VectorXd > requests;
  requests.reserve( logical_candidates.size() * target_sample_count );

  pending_candidates_.clear();
  pending_candidates_.reserve( logical_candidates.size() );

  for ( const Eigen::VectorXd &parameters : logical_candidates )
  {
    CandidateEvaluation aggregate;
    aggregate.parameters = parameters;
    aggregate.status = CandidateEvaluationStatus::NotEvaluated;
    pending_candidates_.push_back( PendingCandidate{ std::move( aggregate ), target_sample_count } );

    // The repeated vectors are one logical candidate evaluated N_k times.
    // Optimizer still sees an ordinary flat batch of parameter vectors.
    for ( std::size_t sample = 0; sample < target_sample_count; ++sample )
    {
      requests.push_back( parameters );
    }
  }

  return requests;
}

std::vector< CandidateEvaluation >
ChangStochasticNelderMeadMethod::aggregatePendingEvaluations( const std::vector< CandidateEvaluation > &evaluations )
{
  if ( !multiple_samples_warning_emitted_ &&
       std::any_of( evaluations.begin(), evaluations.end(),
                    []( const CandidateEvaluation &evaluation ) { return evaluation.evaluation.sampleCount() > 1; } ) )
  {
    std::cerr << "ChangStochasticNelderMeadMethod: each problem evaluation contains multiple samples; "
                 "this method already performs repeated sampling and compares cumulative means\n";
    multiple_samples_warning_emitted_ = true;
  }

  std::size_t expected_evaluation_count = 0;
  for ( const PendingCandidate &pending : pending_candidates_ )
  {
    expected_evaluation_count += pending.requested_evaluations;
  }
  if ( evaluations.size() != expected_evaluation_count )
  {
    throw std::invalid_argument(
      "ChangStochasticNelderMeadMethod: evaluation count does not match the requested samples" );
  }

  std::vector< CandidateEvaluation > aggregated_evaluations;
  aggregated_evaluations.reserve( pending_candidates_.size() );
  std::size_t evaluation_index = 0;

  for ( PendingCandidate &pending : pending_candidates_ )
  {
    for ( std::size_t replication = 0; replication < pending.requested_evaluations; ++replication )
    {
      const CandidateEvaluation &evaluation = evaluations.at( evaluation_index++ );
      if ( evaluation.status == CandidateEvaluationStatus::Succeeded )
      {
        // One CandidateEvaluation is one outer Chang replication. Internal
        // problem samples are represented by their black-box mean.
        pending.aggregate.evaluation.appendSample( evaluation.meanValue() );
        pending.aggregate.evaluation.appendExecutionMetrics( evaluation.evaluation.executionMetrics() );
      }
    }

    pending.aggregate.status = pending.aggregate.evaluation.sampleCount() > 0 ? CandidateEvaluationStatus::Succeeded
                                                                              : CandidateEvaluationStatus::Failed;
    aggregated_evaluations.push_back( std::move( pending.aggregate ) );
  }

  pending_candidates_.clear();
  return aggregated_evaluations;
}

void ChangStochasticNelderMeadMethod::prepareNextIteration()
{
  const std::size_t target_sample_count = targetSampleCount();
  const bool refresh_required = std::any_of(
    simplex().begin(), simplex().end(),
    [&]( const CandidateEvaluation &vertex )
    {
      return vertex.status != CandidateEvaluationStatus::Succeeded ||
        vertex.evaluation.sampleCount() < target_sample_count;
    } );

  if ( refresh_required )
  {
    chang_phase_ = ChangPhase::SimplexRefresh;
    return;
  }

  chang_phase_ = ChangPhase::Delegating;
  prepareIteration();
}

void ChangStochasticNelderMeadMethod::handleSimplexRefresh(
  const std::vector< CandidateEvaluation > &evaluations )
{
  std::vector< CandidateEvaluation > refreshed_simplex =
    aggregatePendingEvaluations( evaluations );

  if ( refreshed_simplex.size() != vertexCount() )
  {
    throw std::logic_error(
      "ChangStochasticNelderMeadMethod: refreshed simplex has an invalid vertex count" );
  }

  for ( std::size_t index = 0; index < refreshed_simplex.size(); ++index )
  {
    simplexVertex( index ) = std::move( refreshed_simplex.at( index ) );
  }

  chang_phase_ = ChangPhase::Delegating;
  prepareIteration();
}

Eigen::VectorXd ChangStochasticNelderMeadMethod::generateAdaptiveRandomCandidate()
{
  if ( unit_distribution_( generator_ ) < hyperparameters_.global_search_probability )
  {
    return generateGlobalAdaptiveRandomCandidate();
  }

  return generateLocalAdaptiveRandomCandidate();
}

Eigen::VectorXd ChangStochasticNelderMeadMethod::generateGlobalAdaptiveRandomCandidate()
{
  Eigen::VectorXd candidate( hyperparameters_.lower_bound.size() );

  for ( Eigen::Index index = 0; index < candidate.size(); ++index )
  {
    std::uniform_real_distribution< double > distribution(
      hyperparameters_.lower_bound[index], hyperparameters_.upper_bound[index] );
    candidate[index] = distribution( generator_ );
  }

  return candidate;
}

Eigen::VectorXd ChangStochasticNelderMeadMethod::generateLocalAdaptiveRandomCandidate()
{
  std::vector< double > fitness_values;
  fitness_values.reserve( simplex().size() );
  double maximum_fitness = 0.0;

  for ( const CandidateEvaluation &vertex : simplex() )
  {
    const double value = vertex.meanValue();
    if ( !std::isfinite( value ) || value <= 0.0 )
    {
      throw ConfigurationEvaluationError(
        "ChangStochasticNelderMeadMethod: local ARS with F(x) = 1 / g(x) requires finite, strictly positive objective values" );
    }

    const double fitness = 1.0 / value;
    if ( !std::isfinite( fitness ) || fitness <= 0.0 )
    {
      throw ConfigurationEvaluationError(
        "ChangStochasticNelderMeadMethod: local ARS with F(x) = 1 / g(x) requires finite, strictly positive fitness values" );
    }

    fitness_values.push_back( fitness );
    maximum_fitness = std::max( maximum_fitness, fitness );
  }

  for ( double &fitness : fitness_values )
  {
    fitness /= maximum_fitness;
  }

  std::discrete_distribution< std::size_t > vertex_distribution(
    fitness_values.begin(), fitness_values.end() );
  const std::size_t selected_index = vertex_distribution( generator_ );
  const Eigen::VectorXd &center = simplex().at( selected_index ).parameters;

  double radius = std::numeric_limits< double >::infinity();
  for ( std::size_t index = 0; index < simplex().size(); ++index )
  {
    if ( index == selected_index )
    {
      continue;
    }
    radius = std::min( radius, ( simplex().at( index ).parameters - center ).norm() );
  }

  if ( !std::isfinite( radius ) || radius <= 0.0 )
  {
    throw ConfigurationEvaluationError(
      "ChangStochasticNelderMeadMethod: local ARS neighborhood is degenerate" );
  }

  std::normal_distribution< double > standard_normal_distribution( 0.0, 1.0 );
  const double inverse_dimension = 1.0 / static_cast< double >( center.size() );

  while ( true )
  {
    Eigen::VectorXd direction( center.size() );
    double direction_norm = 0.0;
    do
    {
      for ( Eigen::Index index = 0; index < direction.size(); ++index )
      {
        direction[index] = standard_normal_distribution( generator_ );
      }
      direction_norm = direction.norm();
    }
    while ( !std::isfinite( direction_norm ) || direction_norm == 0.0 );

    direction /= direction_norm;
    const double sampled_radius = radius * std::pow( unit_distribution_( generator_ ), inverse_dimension );
    const Eigen::VectorXd candidate = center + sampled_radius * direction;

    if ( candidate.allFinite() &&
         ( candidate.array() >= hyperparameters_.lower_bound.array() ).all() &&
         ( candidate.array() <= hyperparameters_.upper_bound.array() ).all() )
    {
      return candidate;
    }
  }
}

std::vector< Eigen::VectorXd > ChangStochasticNelderMeadMethod::ask()
{
  if ( chang_phase_ == ChangPhase::WaitingSimplexRefresh ||
       chang_phase_ == ChangPhase::WaitingAdaptiveRandomSearch )
  {
    throw std::logic_error(
      "ChangStochasticNelderMeadMethod: ask called while waiting for evaluations" );
  }

  if ( chang_phase_ == ChangPhase::Delegating && phase() == Phase::InitialSimplex )
  {
    chang_phase_ = ChangPhase::SimplexRefresh;
  }

  if ( chang_phase_ == ChangPhase::Delegating && phase() == Phase::Shrink )
  {
    // Chang replaces the ordinary shrink with one local or global ARS proposal.
    chang_phase_ = ChangPhase::AdaptiveRandomSearch;
  }

  if ( chang_phase_ == ChangPhase::SimplexRefresh )
  {
    std::vector< Eigen::VectorXd > requests = requestSimplexRefresh();
    if ( requests.empty() )
    {
      chang_phase_ = ChangPhase::Delegating;
      prepareIteration();
      return ask();
    }

    chang_phase_ = ChangPhase::WaitingSimplexRefresh;
    return requests;
  }

  if ( chang_phase_ == ChangPhase::AdaptiveRandomSearch )
  {
    chang_phase_ = ChangPhase::WaitingAdaptiveRandomSearch;
    return requestNewCandidates( { generateAdaptiveRandomCandidate() } );
  }

  // The base method constructs each geometric proposal once; this wrapper
  // expands every proposal into N_k repeated evaluations.
  return requestNewCandidates( NelderMeadMethod::ask() );
}

OptimizerMethodUpdate ChangStochasticNelderMeadMethod::handleAdaptiveRandomSearch(
  const std::vector< CandidateEvaluation > &evaluations )
{
  std::vector< CandidateEvaluation > aggregated_evaluations = aggregatePendingEvaluations( evaluations );
  const CandidateEvaluation &candidate = aggregated_evaluations.front();

  if ( candidate.status == CandidateEvaluationStatus::Succeeded &&
       candidate.meanValue() <= simplex().back().meanValue() )
  {
    completeIteration( candidate );
    return {};
  }

  chang_phase_ = ChangPhase::AdaptiveRandomSearch;
  return OptimizerMethodUpdate{ false };
}

OptimizerMethodUpdate ChangStochasticNelderMeadMethod::tellImpl(
  const std::vector< CandidateEvaluation > &evaluations )
{
  if ( chang_phase_ == ChangPhase::WaitingSimplexRefresh )
  {
    handleSimplexRefresh( evaluations );
    return OptimizerMethodUpdate{ false };
  }
  if ( chang_phase_ == ChangPhase::WaitingAdaptiveRandomSearch )
  {
    return handleAdaptiveRandomSearch( evaluations );
  }
  if ( chang_phase_ != ChangPhase::Delegating )
  {
    throw std::logic_error( "ChangStochasticNelderMeadMethod: tell called without pending evaluations" );
  }

  std::vector< CandidateEvaluation > aggregated_evaluations = aggregatePendingEvaluations( evaluations );

  if ( phase() == Phase::WaitingOutsideContraction )
  {
    const CandidateEvaluation &outside_contraction = aggregated_evaluations.front();
    const bool contraction_failed = outside_contraction.status != CandidateEvaluationStatus::Succeeded ||
      outside_contraction.meanValue() >= reflectedCandidate().meanValue();

    if ( contraction_failed )
    {
      // Chang invokes ARS when either contraction fails. Mapping the failure
      // to Shrink lets ask() intercept it without duplicating the base flow.
      setPhase( Phase::Shrink );
      return OptimizerMethodUpdate{ false };
    }
  }

  const OptimizerMethodUpdate update =
    NelderMeadMethod::tellImpl( aggregated_evaluations );
  return update;
}

void ChangStochasticNelderMeadMethod::updateBestCandidate(
  const std::vector< CandidateEvaluation > & )
{
  const std::optional< CandidateEvaluation > current_best =
    extractBestCandidate( simplex() );

  if ( current_best.has_value() )
  {
    // The current simplex contains Chang's cumulative estimates, while the
    // evaluations received by tell() are only individual outer replications.
    best_candidate_evaluation_ = *current_best;
  }
}

void ChangStochasticNelderMeadMethod::reset()
{
  NelderMeadMethod::reset();
  chang_phase_ = ChangPhase::Delegating;
  pending_candidates_.clear();
  generator_.seed( hyperparameters_.random_seed );
  unit_distribution_.reset();
  multiple_samples_warning_emitted_ = false;
}
