#include "optimizations_methods/CMAESMethod.hpp"
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>
#include <stdexcept>
#include "ConfigurationEvaluationError.hpp"

namespace
{
  /**
   * @brief Resizes and fills a vector batch with independent standard-normal components.
   * @param[in,out] samples Batch storage to populate in order.
   * @param[in] dimension Component count of every sample.
   * @param[in,out] rng Random engine advanced by all generated components.
   * @param[in,out] normal Reusable standard-normal distribution.
   */
  void fillStandardNormal( std::vector< Eigen::VectorXd > &samples, Eigen::Index dimension, std::mt19937_64 &rng,
                           std::normal_distribution< double > &normal )
  {
    for ( Eigen::VectorXd &sample : samples )
    {
      sample.resize( dimension );
      double *values = sample.data();

      for ( Eigen::Index index = 0; index < dimension; ++index )
      {
        values[index] = normal( rng );
      }
    }
  }
} // namespace

CMAESMethod::CMAESMethod( std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy ) :
    CMAESMethod( CMAESHyperparameters{}, number_of_parameters, initial_parameters_strategy )
{
}


CMAESMethod::CMAESMethod( const CMAESHyperparameters &hyperparameters, std::size_t number_of_parameters,
                          InitialParametersStrategy *initial_parameters_strategy ) :
    OptimizerMethod( initial_parameters_strategy ), hyperparameters_( hyperparameters ),
    dimension_( number_of_parameters ), population_size_( hyperparameters.population_size ),
    selected_parent_count_( hyperparameters.population_size / 2 ),
    recombination_weights_(
      Eigen::VectorXd::Zero( static_cast< Eigen::Index >( hyperparameters.population_size / 2 ) ) ),
    effective_selection_mass_( 0.0 ), mean_learning_rate_( 1.0 ), step_size_path_learning_rate_( 0.0 ),
    step_size_damping_( 0.0 ), covariance_path_learning_rate_( 0.0 ), rank_one_learning_rate_( 0.0 ),
    rank_mu_learning_rate_( 0.0 ), expected_normal_norm_( 0.0 ),
    initial_mean_(), mean_( static_cast< Eigen::Index >( number_of_parameters ) ),
    global_step_size_( hyperparameters.initial_sigma ),
    covariance_matrix_( Eigen::MatrixXd::Identity( static_cast< Eigen::Index >( number_of_parameters ),
                                                   static_cast< Eigen::Index >( number_of_parameters ) ) ),
    step_size_path_( Eigen::VectorXd::Zero( static_cast< Eigen::Index >( number_of_parameters ) ) ),
    covariance_path_( Eigen::VectorXd::Zero( static_cast< Eigen::Index >( number_of_parameters ) ) ),
    eigenvectors_( static_cast< Eigen::Index >( number_of_parameters ),
                   static_cast< Eigen::Index >( number_of_parameters ) ),
    axis_scaling_( static_cast< Eigen::Index >( number_of_parameters ),
                   static_cast< Eigen::Index >( number_of_parameters ) ),
    generator_( hyperparameters.random_seed ), standard_normal_distribution_( 0.0, 1.0 )
{
  validateConfiguration();

  initial_mean_ = initial_parameters_strategy_->generateInitialParameters();
  if ( initial_mean_.size() != static_cast< Eigen::Index >( dimension_ ) )
  {
    throw std::invalid_argument( "CMAESMethod: initial mean dimension does not match the search dimension" );
  }
  if ( !initial_mean_.allFinite() )
  {
    throw std::invalid_argument( "CMAESMethod: initial mean values must be finite" );
  }

  initializeStrategyParameters();

  standard_normal_vectors_.reserve( population_size_ );
  transformed_vectors_.reserve( population_size_ );
  population_.reserve( population_size_ );

  initializeState();
}


void CMAESMethod::validateConfiguration() const
{
  if ( initial_parameters_strategy_ == nullptr )
  {
    throw std::invalid_argument( "CMAESMethod: initial parameters strategy cannot be null" );
  }
  if ( dimension_ == 0 )
  {
    throw std::invalid_argument( "CMAESMethod: number of parameters must be greater than zero" );
  }
  if ( population_size_ < 2 )
  {
    throw std::invalid_argument( "CMAESMethod: population size must be at least two" );
  }
  if ( !std::isfinite( global_step_size_ ) || global_step_size_ <= 0.0 )
  {
    throw std::invalid_argument( "CMAESMethod: initial sigma must be finite and positive" );
  }
}


void CMAESMethod::initializeStrategyParameters()
{
  for ( std::size_t index = 0; index < selected_parent_count_; ++index )
  {
    recombination_weights_( static_cast< Eigen::Index >( index ) ) =
      std::log( ( static_cast< double >( population_size_ ) + 1.0 ) / 2.0 ) -
      std::log( static_cast< double >( index + 1 ) );
  }
  recombination_weights_ /= recombination_weights_.sum();
  effective_selection_mass_ = 1.0 / recombination_weights_.squaredNorm();

  const double dimension = static_cast< double >( dimension_ );
  step_size_path_learning_rate_ = ( effective_selection_mass_ + 2.0 ) / ( dimension + effective_selection_mass_ + 5.0 );
  step_size_damping_ = 1.0 +
    2.0 * std::max( 0.0, std::sqrt( ( effective_selection_mass_ - 1.0 ) / ( dimension + 1.0 ) ) - 1.0 ) +
    step_size_path_learning_rate_;
  covariance_path_learning_rate_ =
    ( 4.0 + effective_selection_mass_ / dimension ) / ( dimension + 4.0 + 2.0 * effective_selection_mass_ / dimension );
  rank_one_learning_rate_ = 2.0 / ( ( dimension + 1.3 ) * ( dimension + 1.3 ) + effective_selection_mass_ );
  rank_mu_learning_rate_ =
    std::min( 1.0 - rank_one_learning_rate_,
              2.0 * ( 0.25 + effective_selection_mass_ + 1.0 / effective_selection_mass_ - 2.0 ) /
                ( ( dimension + 2.0 ) * ( dimension + 2.0 ) + effective_selection_mass_ ) );
  expected_normal_norm_ =
    std::sqrt( dimension ) * ( 1.0 - 1.0 / ( 4.0 * dimension ) + 1.0 / ( 21.0 * dimension * dimension ) );
}


void CMAESMethod::initializeState()
{
  const Eigen::Index dimension = static_cast< Eigen::Index >( dimension_ );

  mean_ = initial_mean_;
  global_step_size_ = hyperparameters_.initial_sigma;
  covariance_matrix_.setIdentity( dimension, dimension );
  step_size_path_.setZero( dimension );
  covariance_path_.setZero( dimension );
  eigenvectors_.setIdentity( dimension, dimension );
  axis_scaling_.setIdentity( dimension, dimension );

  standard_normal_vectors_.clear();
  transformed_vectors_.clear();
  population_.clear();

  generator_.seed( hyperparameters_.random_seed );
  standard_normal_distribution_.reset();
  generation_ = 0;
}

std::vector< Eigen::VectorXd > CMAESMethod::ask()
{
  sampleStandardNormalVectors();
  transformStandardNormalVectors();
  constructPopulation();

  return population_;
}


void CMAESMethod::sampleStandardNormalVectors()
{
  standard_normal_vectors_.resize( population_size_ );
  fillStandardNormal( standard_normal_vectors_, static_cast< Eigen::Index >( dimension_ ), generator_,
                      standard_normal_distribution_ );
}


void CMAESMethod::transformStandardNormalVectors()
{
  transformed_vectors_.clear();
  for ( const Eigen::VectorXd &z_k : standard_normal_vectors_ )
  {
    transformed_vectors_.emplace_back( eigenvectors_ * axis_scaling_ * z_k );
  }
}


void CMAESMethod::constructPopulation()
{
  population_.clear();
  for ( const Eigen::VectorXd &y_k : transformed_vectors_ )
  {
    population_.emplace_back( mean_ + global_step_size_ * y_k );
  }
}


OptimizerMethodUpdate CMAESMethod::tellImpl( const std::vector< CandidateEvaluation > &evaluations )
{
  const std::vector< CandidateEvaluation > sorted_candidates = normalizeAndSortSuccessfulEvaluations( evaluations );

  if ( sorted_candidates.size() < selected_parent_count_ )
  {
    return {};
  }

  const Eigen::VectorXd weighted_mean_step = calculateWeightedMeanStep( sorted_candidates );

  updateMean( weighted_mean_step );
  updateStepSizePath( weighted_mean_step );
  updateGlobalStepSize();

  const double h_sigma = calculateHSigma();
  updateCovariancePath( weighted_mean_step, h_sigma );
  updateCovarianceMatrix( sorted_candidates, h_sigma );
  updateEigendecomposition();

  ++generation_;
  return {};
}


std::vector< CandidateEvaluation >
CMAESMethod::normalizeAndSortSuccessfulEvaluations( const std::vector< CandidateEvaluation > &evaluations ) const
{
  if ( evaluations.size() != population_size_ )
  {
    throw std::invalid_argument( "CMAESMethod: tell requires one evaluation for every sampled candidate" );
  }

  std::vector< CandidateEvaluation > sorted_candidates = evaluations;

  for ( CandidateEvaluation &candidate : sorted_candidates )
  {
    if ( candidate.parameters.size() != static_cast< Eigen::Index >( dimension_ ) )
    {
      throw std::invalid_argument( "CMAESMethod: candidate dimension does not match the search dimension" );
    }

    candidate.parameters = ( candidate.parameters - mean_ ) / global_step_size_;
  }

  std::erase_if( sorted_candidates, []( const CandidateEvaluation &candidate )
                 { return candidate.status != CandidateEvaluationStatus::Succeeded; } );

  std::sort( sorted_candidates.begin(), sorted_candidates.end(),
             []( const CandidateEvaluation &left, const CandidateEvaluation &right )
             { return left.meanValue() < right.meanValue(); } );

  return sorted_candidates;
}


Eigen::VectorXd
CMAESMethod::calculateWeightedMeanStep( const std::vector< CandidateEvaluation > &sorted_candidates ) const
{
  Eigen::VectorXd weighted_mean_step = Eigen::VectorXd::Zero( static_cast< Eigen::Index >( dimension_ ) );

  for ( std::size_t index = 0; index < selected_parent_count_; ++index )
  {
    weighted_mean_step +=
      recombination_weights_( static_cast< Eigen::Index >( index ) ) * sorted_candidates.at( index ).parameters;
  }

  return weighted_mean_step;
}


void CMAESMethod::updateMean( const Eigen::VectorXd &weighted_mean_step )
{
  mean_ += mean_learning_rate_ * global_step_size_ * weighted_mean_step;
}


void CMAESMethod::updateStepSizePath( const Eigen::VectorXd &weighted_mean_step )
{
  // C^(-1/2) y_w = B D^(-1) B^T y_w.
  Eigen::VectorXd whitened_mean_step = eigenvectors_.transpose() * weighted_mean_step;
  whitened_mean_step.array() /= axis_scaling_.diagonal().array();
  whitened_mean_step = eigenvectors_ * whitened_mean_step;

  step_size_path_ = ( 1.0 - step_size_path_learning_rate_ ) * step_size_path_ +
    std::sqrt( step_size_path_learning_rate_ * ( 2.0 - step_size_path_learning_rate_ ) * effective_selection_mass_ ) *
      whitened_mean_step;
}


void CMAESMethod::updateGlobalStepSize()
{
  global_step_size_ *= std::exp( step_size_path_learning_rate_ / step_size_damping_ *
                                 ( step_size_path_.norm() / expected_normal_norm_ - 1.0 ) );
}


double CMAESMethod::calculateHSigma() const
{
  const double completed_generation_count = static_cast< double >( generation_ + 1 );
  const double path_length_normalization =
    std::sqrt( 1.0 - std::pow( 1.0 - step_size_path_learning_rate_, 2.0 * completed_generation_count ) );
  const double path_length_threshold =
    ( 1.4 + 2.0 / ( static_cast< double >( dimension_ ) + 1.0 ) ) * expected_normal_norm_;

  return step_size_path_.norm() / path_length_normalization < path_length_threshold ? 1.0 : 0.0;
}


void CMAESMethod::updateCovariancePath( const Eigen::VectorXd &weighted_mean_step, double h_sigma )
{
  covariance_path_ = ( 1.0 - covariance_path_learning_rate_ ) * covariance_path_ +
    h_sigma *
      std::sqrt( covariance_path_learning_rate_ * ( 2.0 - covariance_path_learning_rate_ ) *
                 effective_selection_mass_ ) *
      weighted_mean_step;
}


void CMAESMethod::updateCovarianceMatrix( const std::vector< CandidateEvaluation > &sorted_candidates, double h_sigma )
{
  Eigen::MatrixXd rank_mu_matrix =
    Eigen::MatrixXd::Zero( static_cast< Eigen::Index >( dimension_ ), static_cast< Eigen::Index >( dimension_ ) );

  for ( std::size_t index = 0; index < selected_parent_count_; ++index )
  {
    const Eigen::VectorXd &selected_step = sorted_candidates.at( index ).parameters;
    rank_mu_matrix +=
      recombination_weights_( static_cast< Eigen::Index >( index ) ) * selected_step * selected_step.transpose();
  }

  const double covariance_retention = 1.0 - rank_one_learning_rate_ - rank_mu_learning_rate_ +
    ( 1.0 - h_sigma ) * rank_one_learning_rate_ * covariance_path_learning_rate_ *
      ( 2.0 - covariance_path_learning_rate_ );

  covariance_matrix_ = covariance_retention * covariance_matrix_ +
    rank_one_learning_rate_ * covariance_path_ * covariance_path_.transpose() + rank_mu_learning_rate_ * rank_mu_matrix;
}


void CMAESMethod::updateEigendecomposition()
{
  covariance_matrix_ = ( 0.5 * ( covariance_matrix_ + covariance_matrix_.transpose() ) ).eval();

  const Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > decomposition( covariance_matrix_ );
  if ( decomposition.info() != Eigen::Success || !decomposition.eigenvalues().allFinite() ||
       decomposition.eigenvalues().minCoeff() <= 0.0 )
  {
    throw ConfigurationEvaluationError( "CMAESMethod: covariance eigendecomposition failed" );
  }

  eigenvectors_ = decomposition.eigenvectors();
  axis_scaling_ = decomposition.eigenvalues().cwiseSqrt().asDiagonal();
}


void CMAESMethod::reset()
{
  OptimizerMethod::reset();
  initializeState();
}
