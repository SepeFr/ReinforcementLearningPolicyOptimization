#include "surrogate_models/SurrogateModel.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/QR>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include "CandidateEvaluation.hpp"

LocalQuadraticSurrogate::LocalQuadraticSurrogate( LocalQuadraticSurrogateConfiguration configuration ) :
    configuration_( std::move( configuration ) )
{
  if ( !std::isfinite( configuration_.local_radius_multiplier ) || configuration_.local_radius_multiplier <= 0.0 )
  {
    throw std::invalid_argument( "LocalQuadraticSurrogate: local radius multiplier must be finite and positive" );
  }
  if ( !std::isfinite( configuration_.curvature_regularization ) || configuration_.curvature_regularization < 0.0 )
  {
    throw std::invalid_argument( "LocalQuadraticSurrogate: curvature regularization must be finite and non-negative" );
  }
}


Eigen::VectorXd LocalQuadraticSurrogate::makeFeatures( Eigen::Ref< const Eigen::VectorXd > scaled_parameters ) const
{
  const Eigen::Index parameter_count = scaled_parameters.size();
  const Eigen::Index coefficient_count = 1 + parameter_count + parameter_count * ( parameter_count + 1 ) / 2;

  Eigen::VectorXd features( coefficient_count );
  Eigen::Index feature_index = 0;

  // Intercept.
  features( feature_index++ ) = 1.0;

  // Linear terms: g_j * s_j.
  for ( Eigen::Index j = 0; j < parameter_count; ++j )
  {
    features( feature_index++ ) = scaled_parameters( j );
  }

  // Diagonal quadratic terms: 1/2 * H_jj * s_j^2.
  for ( Eigen::Index j = 0; j < parameter_count; ++j )
  {
    features( feature_index++ ) = 0.5 * scaled_parameters( j ) * scaled_parameters( j );
  }

  // Mixed quadratic terms: H_jl * s_j * s_l for j < l.
  for ( Eigen::Index j = 0; j < parameter_count; ++j )
  {
    for ( Eigen::Index l = j + 1; l < parameter_count; ++l )
    {
      features( feature_index++ ) = scaled_parameters( j ) * scaled_parameters( l );
    }
  }

  assert( feature_index == coefficient_count );
  return features;
}


Eigen::MatrixXd LocalQuadraticSurrogate::makeRegularizationMatrix( Eigen::Index parameter_count ) const
{
  const Eigen::Index curvature_coefficient_count = parameter_count * ( parameter_count + 1 ) / 2;
  const Eigen::Index coefficient_count = 1 + parameter_count + curvature_coefficient_count;

  Eigen::MatrixXd regularization_matrix = Eigen::MatrixXd::Zero( curvature_coefficient_count, coefficient_count );

  Eigen::Index row = 0;
  Eigen::Index coefficient = 1 + parameter_count;

  // Diagonal H_jj coefficients.
  for ( Eigen::Index j = 0; j < parameter_count; ++j )
  {
    regularization_matrix( row++, coefficient++ ) = 1.0;
  }

  // H_jl coefficients for j < l appear twice in the Frobenius norm.
  for ( Eigen::Index j = 0; j < parameter_count; ++j )
  {
    for ( Eigen::Index l = j + 1; l < parameter_count; ++l )
    {
      regularization_matrix( row++, coefficient++ ) = std::sqrt( 2.0 );
    }
  }

  assert( row == curvature_coefficient_count );
  assert( coefficient == coefficient_count );
  return regularization_matrix;
}


bool LocalQuadraticSurrogate::fit( const std::vector< CandidateEvaluation > &history,
                                   Eigen::Ref< const Eigen::VectorXd > center, double scale )
{
  if ( !std::isfinite( scale ) || scale <= 0.0 || !center.allFinite() )
  {
    return false;
  }

  const double local_radius = configuration_.local_radius_multiplier * scale;
  const double squared_local_radius = local_radius * local_radius;

  std::vector< CandidateEvaluation > filtered_history;
  filtered_history.reserve( history.size() );
  std::copy_if( history.begin(), history.end(), std::back_inserter( filtered_history ),
                [&]( const CandidateEvaluation &candidate )
                {
                  return candidate.status == CandidateEvaluationStatus::Succeeded &&
                    candidate.parameters.size() == center.size() &&
                    ( candidate.parameters - center ).squaredNorm() <= squared_local_radius;
                } );

  const std::size_t parameter_count = static_cast< std::size_t >( center.size() );
  const std::size_t coefficient_count = 1 + parameter_count + parameter_count * ( parameter_count + 1 ) / 2;
  if ( filtered_history.size() < coefficient_count )
  {
    return false;
  }

  const Eigen::Index observation_count = static_cast< Eigen::Index >( filtered_history.size() );

  Eigen::MatrixXd design_matrix( observation_count, coefficient_count );
  Eigen::VectorXd values( observation_count );

  for ( Eigen::Index candidate_row = 0; candidate_row < observation_count; candidate_row++ )
  {
    const CandidateEvaluation &candidate = filtered_history.at( static_cast< std::size_t >( candidate_row ) );
    const Eigen::VectorXd scaled_parameters = ( candidate.parameters - center ) / scale;
    design_matrix.row( candidate_row ) = makeFeatures( scaled_parameters ).transpose();

    values( candidate_row ) = candidate.meanValue();
  }

  const Eigen::MatrixXd regularization_matrix =
    makeRegularizationMatrix( static_cast< Eigen::Index >( parameter_count ) );
  const Eigen::Index regularization_row_count = regularization_matrix.rows();


  Eigen::MatrixXd augmented_design_matrix( observation_count + regularization_row_count,
                                           static_cast< Eigen::Index >( coefficient_count ) );

  augmented_design_matrix.topRows( observation_count ) = design_matrix;
  augmented_design_matrix.bottomRows( regularization_row_count ) =
    std::sqrt( configuration_.curvature_regularization ) * regularization_matrix;

  Eigen::VectorXd augmented_values = Eigen::VectorXd::Zero( observation_count + regularization_row_count );
  augmented_values.head( observation_count ) = values;

  Eigen::ColPivHouseholderQR< Eigen::MatrixXd > qr( augmented_design_matrix );

  if ( static_cast< std::size_t >( qr.rank() ) < coefficient_count )
  {
    return false;
  }

  const Eigen::VectorXd fitted_coefficients = qr.solve( augmented_values );
  if ( !fitted_coefficients.allFinite() )
  {
    return false;
  }

  center_ = center;
  scale_ = scale;
  coefficients_ = fitted_coefficients;
  return true;
}


double LocalQuadraticSurrogate::predict( Eigen::Ref< const Eigen::VectorXd > parameters ) const
{
  if ( coefficients_.size() == 0 )
  {
    throw std::logic_error( "LocalQuadraticSurrogate: the model must be fitted before prediction" );
  }
  if ( parameters.size() != center_.size() )
  {
    throw std::invalid_argument( "LocalQuadraticSurrogate: parameter count does not match the fitted model" );
  }

  const Eigen::VectorXd scaled_parameters = ( parameters - center_ ) / scale_;

  const Eigen::VectorXd features = makeFeatures( scaled_parameters );

  return features.dot( coefficients_ );
}


void LocalQuadraticSurrogate::reset()
{
  center_.resize( 0 );
  scale_ = 1.0;
  coefficients_.resize( 0 );
}


RBFSurrogate::RBFSurrogate( RBFSurrogateConfiguration configuration ) : configuration_( std::move( configuration ) )
{
  if ( !std::isfinite( configuration_.regularization ) || configuration_.regularization < 0.0 )
  {
    throw std::invalid_argument( "RBFSurrogate: regularization must be finite and non-negative" );
  }
  if ( !std::isfinite( configuration_.local_radius_multiplier ) || configuration_.local_radius_multiplier <= 0.0 )
  {
    throw std::invalid_argument( "RBFSurrogate: local radius multiplier must be finite and positive" );
  }
  if ( configuration_.maximum_coreset_size == 0 )
  {
    throw std::invalid_argument( "RBFSurrogate: maximum coreset size must be greater than zero" );
  }

  switch ( configuration_.function )
  {
    case RBFFunction::Cubic:
    case RBFFunction::Linear:
      break;

    default:
      throw std::invalid_argument( "RBFSurrogate: unknown radial basis function" );
  }
}


std::vector< std::size_t >
RBFSurrogate::selectAffinelyIndependentCandidates( const std::vector< CandidateEvaluation > &candidates,
                                                   Eigen::Ref< const Eigen::VectorXd > center, double scale ) const
{
  std::vector< std::size_t > selected_indices;
  if ( candidates.empty() )
  {
    return selected_indices;
  }

  // Use the sample closest to the current center as the affine anchor.
  std::size_t anchor_index = 0;
  for ( std::size_t index = 1; index < candidates.size(); ++index )
  {
    if ( ( candidates.at( index ).parameters - center ).squaredNorm() <
         ( candidates.at( anchor_index ).parameters - center ).squaredNorm() )
    {
      anchor_index = index;
    }
  }

  selected_indices.push_back( anchor_index );
  std::vector< bool > selected( candidates.size(), false );
  selected.at( anchor_index ) = true;

  // Residuals are displacements from the anchor, expressed in local coordinates.
  std::vector< Eigen::VectorXd > orthogonal_residuals;
  orthogonal_residuals.reserve( candidates.size() );
  const Eigen::VectorXd &anchor = candidates.at( anchor_index ).parameters;
  for ( const CandidateEvaluation &candidate : candidates )
  {
    orthogonal_residuals.emplace_back( ( candidate.parameters - anchor ) / scale );
  }

  const std::size_t parameter_count = static_cast< std::size_t >( center.size() );
  const double squared_tolerance = std::numeric_limits< double >::epsilon();

  for ( std::size_t direction_count = 0; direction_count < parameter_count; ++direction_count )
  {
    std::size_t best_index = candidates.size();
    double best_squared_residual = squared_tolerance;

    for ( std::size_t index = 0; index < candidates.size(); ++index )
    {
      if ( selected.at( index ) )
      {
        continue;
      }

      const double squared_residual = orthogonal_residuals.at( index ).squaredNorm();
      if ( squared_residual > best_squared_residual )
      {
        best_squared_residual = squared_residual;
        best_index = index;
      }
    }

    if ( best_index == candidates.size() )
    {
      break;
    }

    const Eigen::VectorXd new_direction = orthogonal_residuals.at( best_index ) / std::sqrt( best_squared_residual );
    selected.at( best_index ) = true;
    selected_indices.push_back( best_index );

    // Modified Gram-Schmidt removes the new direction from every remaining residual.
    for ( std::size_t index = 0; index < candidates.size(); ++index )
    {
      if ( !selected.at( index ) )
      {
        orthogonal_residuals.at( index ) -= new_direction * new_direction.dot( orthogonal_residuals.at( index ) );
      }
    }
  }

  return selected_indices;
}


void RBFSurrogate::completeCoresetWithMaximin( const std::vector< CandidateEvaluation > &candidates,
                                               std::vector< std::size_t > &selected_indices,
                                               std::size_t target_count ) const
{
  target_count = std::min( target_count, candidates.size() );
  if ( selected_indices.empty() || selected_indices.size() >= target_count )
  {
    return;
  }

  std::vector< bool > selected( candidates.size(), false );
  std::vector< double > minimum_squared_distances( candidates.size(), std::numeric_limits< double >::infinity() );

  for ( const std::size_t selected_index : selected_indices )
  {
    selected.at( selected_index ) = true;
    for ( std::size_t index = 0; index < candidates.size(); ++index )
    {
      if ( !selected.at( index ) )
      {
        const double squared_distance =
          ( candidates.at( index ).parameters - candidates.at( selected_index ).parameters ).squaredNorm();
        minimum_squared_distances.at( index ) = std::min( minimum_squared_distances.at( index ), squared_distance );
      }
    }
  }

  while ( selected_indices.size() < target_count )
  {
    std::size_t farthest_index = candidates.size();
    double farthest_squared_distance = -1.0;

    for ( std::size_t index = 0; index < candidates.size(); ++index )
    {
      if ( !selected.at( index ) && minimum_squared_distances.at( index ) > farthest_squared_distance )
      {
        farthest_squared_distance = minimum_squared_distances.at( index );
        farthest_index = index;
      }
    }

    if ( farthest_index == candidates.size() )
    {
      break;
    }

    selected.at( farthest_index ) = true;
    selected_indices.push_back( farthest_index );

    for ( std::size_t index = 0; index < candidates.size(); ++index )
    {
      if ( !selected.at( index ) )
      {
        const double squared_distance =
          ( candidates.at( index ).parameters - candidates.at( farthest_index ).parameters ).squaredNorm();
        minimum_squared_distances.at( index ) = std::min( minimum_squared_distances.at( index ), squared_distance );
      }
    }
  }
}


double RBFSurrogate::radialBasisValue( double radius ) const
{
  switch ( configuration_.function )
  {
    case RBFFunction::Cubic:
      return radius * radius * radius;

    case RBFFunction::Linear:
      return radius;
  }

  throw std::logic_error( "RBFSurrogate: unknown radial basis function" );
}


bool RBFSurrogate::fit( const std::vector< CandidateEvaluation > &history, Eigen::Ref< const Eigen::VectorXd > center,
                        double scale )
{
  if ( !std::isfinite( scale ) || scale <= 0.0 || !center.allFinite() )
  {
    return false;
  }

  const std::size_t vector_size = center.size();
  const double local_radius = configuration_.local_radius_multiplier * scale;
  const double squared_local_radius = local_radius * local_radius;

  std::vector< CandidateEvaluation > filtered_history;
  filtered_history.reserve( history.size() );
  std::copy_if( history.begin(), history.end(), std::back_inserter( filtered_history ),
                [&]( const CandidateEvaluation &candidate )
                {
                  return candidate.status == CandidateEvaluationStatus::Succeeded &&
                    candidate.parameters.size() == center.size() &&
                    ( candidate.parameters - center ).squaredNorm() <= squared_local_radius;
                } );

  const std::size_t required_affine_point_count = vector_size + 1;
  if ( filtered_history.size() < required_affine_point_count ||
       configuration_.maximum_coreset_size < required_affine_point_count )
  {
    return false;
  }

  std::vector< std::size_t > selected_indices = selectAffinelyIndependentCandidates( filtered_history, center, scale );
  if ( selected_indices.size() < required_affine_point_count )
  {
    return false;
  }

  // Preserve the affine core, then cover the remaining local space with maximin.
  const std::size_t target_count = std::min( configuration_.maximum_coreset_size, filtered_history.size() );
  completeCoresetWithMaximin( filtered_history, selected_indices, target_count );

  std::vector< CandidateEvaluation > coreset;
  coreset.reserve( selected_indices.size() );
  for ( const std::size_t index : selected_indices )
  {
    coreset.push_back( filtered_history.at( index ) );
  }

  const std::size_t number_of_observations = coreset.size();
  std::vector< CandidateEvaluation > scaled_vectors;
  scaled_vectors.reserve( number_of_observations );
  for ( const CandidateEvaluation &candidate : coreset )
  {
    CandidateEvaluation scaled_candidate = candidate;
    scaled_candidate.parameters = ( scaled_candidate.parameters - center ) / scale;
    scaled_vectors.emplace_back( std::move( scaled_candidate ) );
  }


  Eigen::MatrixXd Phi( number_of_observations, number_of_observations );
  for ( std::size_t i = 0; i < number_of_observations; i++ )
  {
    for ( std::size_t j = 0; j < number_of_observations; j++ )
    {
      Phi( i, j ) =
        radialBasisValue( ( scaled_vectors.at( i ).parameters - scaled_vectors.at( j ).parameters ).norm() );
    }
  }

  Phi.diagonal().array() += configuration_.regularization;

  Eigen::VectorXd values( number_of_observations + vector_size + 1 );
  values.setZero();

  Eigen::MatrixXd P( number_of_observations, vector_size + 1 );
  for ( std::size_t observation_index = 0; observation_index < number_of_observations; ++observation_index )
  {
    P( observation_index, 0 ) = 1.0;
    P.row( observation_index ).tail( vector_size ) = scaled_vectors.at( observation_index ).parameters.transpose();

    values( observation_index ) = scaled_vectors.at( observation_index ).meanValue();
  }

  const std::size_t system_size = number_of_observations + vector_size + 1;

  Eigen::MatrixXd system_matrix = Eigen::MatrixXd::Zero( system_size, system_size );
  system_matrix.topLeftCorner( number_of_observations, number_of_observations ) = Phi;
  system_matrix.topRightCorner( number_of_observations, vector_size + 1 ) = P;
  system_matrix.bottomLeftCorner( vector_size + 1, number_of_observations ) = P.transpose();

  Eigen::FullPivLU< Eigen::MatrixXd > decomposition( system_matrix );
  if ( !decomposition.isInvertible() )
  {
    return false;
  }

  const Eigen::VectorXd solution = decomposition.solve( values );
  if ( !solution.allFinite() )
  {
    return false;
  }

  radial_coefficients_ = solution.head( number_of_observations );
  polynomial_coefficients_ = solution.tail( vector_size + 1 );

  center_ = center;
  scale_ = scale;
  scaled_centers_.clear();
  scaled_centers_.reserve( number_of_observations );
  for ( const CandidateEvaluation &candidate : scaled_vectors )
  {
    scaled_centers_.emplace_back( candidate.parameters );
  }

  return true;
}


double RBFSurrogate::predict( Eigen::Ref< const Eigen::VectorXd > parameters ) const
{
  if ( radial_coefficients_.size() == 0 )
  {
    throw std::logic_error( "RBFSurrogate: the model must be fitted before prediction" );
  }
  if ( parameters.size() != center_.size() )
  {
    throw std::invalid_argument( "RBFSurrogate: parameter count does not match the fitted model" );
  }

  const Eigen::VectorXd scaled_vector = ( parameters - center_ ) / scale_;

  Eigen::VectorXd radial_values( scaled_centers_.size() );
  for ( std::size_t center_index = 0; center_index < scaled_centers_.size(); ++center_index )
  {
    const double radius = ( scaled_vector - scaled_centers_.at( center_index ) ).norm();
    radial_values( center_index ) = radialBasisValue( radius );
  }

  Eigen::VectorXd expanded_vector( scaled_vector.size() + 1 );
  expanded_vector << 1.0, scaled_vector;

  return radial_values.dot( radial_coefficients_ ) + expanded_vector.dot( polynomial_coefficients_ );
}


void RBFSurrogate::reset()
{
  center_.resize( 0 );
  scale_ = 1.0;
  scaled_centers_.clear();
  radial_coefficients_.resize( 0 );
  polynomial_coefficients_.resize( 0 );
}
