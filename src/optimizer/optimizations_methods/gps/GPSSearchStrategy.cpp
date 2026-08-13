#include "optimizations_methods/gps/GPSSearchStrategy.hpp"
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace
{
  double binomialCoefficient( std::size_t n, std::size_t k )
  {
    k = std::min( k, n - k );

    double result = 1.0;
    for ( std::size_t i = 1; i <= k; ++i )
    {
      result *= static_cast< double >( n - k + i ) / static_cast< double >( i );
    }

    return result;
  }

  Eigen::VectorXi projectOntoIntegerL1Ball( const Eigen::VectorXi &direction, std::size_t radius )
  {
    const Eigen::Index l1_norm = direction.cwiseAbs().sum();
    if ( l1_norm <= static_cast< Eigen::Index >( radius ) )
    {
      return direction;
    }

    // Scaling and truncation preserve integer coefficients and guarantee
    // that the projected vector has L1 norm no greater than the radius.
    const double scale = static_cast< double >( radius ) / static_cast< double >( l1_norm );
    Eigen::VectorXi projected = ( scale * direction.cast< double >() ).cast< int >();

    if ( projected.cwiseAbs().sum() == 0 )
    {
      Eigen::Index largest_coordinate = 0;
      direction.cwiseAbs().maxCoeff( &largest_coordinate );
      projected( largest_coordinate ) = direction( largest_coordinate ) < 0 ? -1 : 1;
    }

    return projected;
  }
} // namespace

EmptyGPSSearchStrategy::EmptyGPSSearchStrategy( EmptyGPSSearchConfiguration configuration ) :
    configuration_( std::move( configuration ) )
{
}


RandomMeshGPSSearchStrategy::RandomMeshGPSSearchStrategy( RandomMeshGPSSearchConfiguration configuration ) :
    configuration_( std::move( configuration ) ), generator_( configuration_.random_seed )
{
  if ( configuration_.number_of_points > 0 && configuration_.l1_radius == 0 )
  {
    throw std::invalid_argument( "RandomMeshGPSSearchStrategy: L1 radius must be greater than zero" );
  }
}


std::vector< Eigen::VectorXd > RandomMeshGPSSearchStrategy::ask( const CandidateEvaluation &current_candidate,
                                                                 double mesh_size,
                                                                 Eigen::Ref< const Eigen::MatrixXd > directions,
                                                                 std::size_t iteration )
{
  const std::size_t dimension = static_cast< std::size_t >( directions.cols() );

  std::vector< std::pair< std::size_t, std::size_t > > radius_and_nonzero_count;
  std::vector< double > weights;

  // Count how many integer vectors exist for each L1 radius and number of nonzero coordinates.
  for ( std::size_t radius = 1; radius <= configuration_.l1_radius; ++radius )
  {
    for ( std::size_t nonzero_count = 1; nonzero_count <= std::min( dimension, radius ); ++nonzero_count )
    {
      radius_and_nonzero_count.emplace_back( radius, nonzero_count );
      weights.push_back( binomialCoefficient( dimension, nonzero_count ) *
                         binomialCoefficient( radius - 1, nonzero_count - 1 ) *
                         std::pow( 2.0, static_cast< double >( nonzero_count ) ) );
    }
  }

  // Sampling with these weights makes every valid integer vector equally likely.
  std::discrete_distribution< std::size_t > shape_distribution( weights.begin(), weights.end() );
  std::bernoulli_distribution sign_distribution;

  std::vector< Eigen::VectorXd > candidates;
  candidates.reserve( configuration_.number_of_points );

  for ( std::size_t candidate_index = 0; candidate_index < configuration_.number_of_points; ++candidate_index )
  {
    const auto [radius, nonzero_count] = radius_and_nonzero_count.at( shape_distribution( generator_ ) );

    // Randomly select which coordinates of the integer vector will be nonzero.
    std::vector< Eigen::Index > coordinate_indices( dimension );
    std::iota( coordinate_indices.begin(), coordinate_indices.end(), Eigen::Index{ 0 } );
    std::shuffle( coordinate_indices.begin(), coordinate_indices.end(), generator_ );

    // Split the selected L1 radius into positive integer magnitudes.
    std::vector< std::size_t > separators( radius - 1 );
    std::iota( separators.begin(), separators.end(), std::size_t{ 1 } );
    std::shuffle( separators.begin(), separators.end(), generator_ );
    separators.resize( nonzero_count - 1 );
    std::sort( separators.begin(), separators.end() );
    separators.push_back( radius );

    Eigen::VectorXi integer_direction = Eigen::VectorXi::Zero( static_cast< Eigen::Index >( dimension ) );
    std::size_t previous_separator = 0;

    // Assign every magnitude to a selected coordinate with a random sign.
    for ( std::size_t component = 0; component < nonzero_count; ++component )
    {
      const int magnitude = static_cast< int >( separators.at( component ) - previous_separator );
      integer_direction( coordinate_indices.at( component ) ) =
        sign_distribution( generator_ ) ? magnitude : -magnitude;
      previous_separator = separators.at( component );
    }

    // Map the integer direction onto the current GPS mesh.
    candidates.push_back( current_candidate.parameters + mesh_size * directions * integer_direction.cast< double >() );
  }

  static_cast< void >( iteration );
  return candidates;
}


void RandomMeshGPSSearchStrategy::tell( const CandidateEvaluation &current_candidate,
                                        const std::vector< CandidateEvaluation > &evaluations, double mesh_size )
{
  static_cast< void >( current_candidate );
  static_cast< void >( evaluations );
  static_cast< void >( mesh_size );
}


void RandomMeshGPSSearchStrategy::reset() { generator_.seed( configuration_.random_seed ); }


LatinHypercubeMeshGPSSearchStrategy::LatinHypercubeMeshGPSSearchStrategy(
  LatinHypercubeMeshGPSSearchConfiguration configuration ) :
    configuration_( std::move( configuration ) ), sampler_( configuration_.random_seed )
{
  if ( configuration_.number_of_points > 0 && configuration_.l1_radius == 0 )
  {
    throw std::invalid_argument( "LatinHypercubeMeshGPSSearchStrategy: L1 radius must be greater than zero" );
  }
  if ( configuration_.number_of_points > 0 && configuration_.maximum_batches == 0 )
  {
    throw std::invalid_argument( "LatinHypercubeMeshGPSSearchStrategy: maximum batches must be greater than zero" );
  }
}


std::vector< Eigen::VectorXd > LatinHypercubeMeshGPSSearchStrategy::ask(
  const CandidateEvaluation &current_candidate, double mesh_size, Eigen::Ref< const Eigen::MatrixXd > directions,
  std::size_t iteration )
{
  if ( configuration_.number_of_points == 0 )
  {
    return {};
  }

  const std::size_t coefficient_count = static_cast< std::size_t >( directions.cols() );
  const std::size_t radius = configuration_.l1_radius;
  const std::size_t level_count = 2 * radius + 1;

  std::vector< Eigen::VectorXd > candidates;
  candidates.reserve( configuration_.number_of_points );
  std::set< std::vector< int > > used_directions;

  for ( std::size_t batch = 0;
        batch < configuration_.maximum_batches && candidates.size() < configuration_.number_of_points; ++batch )
  {
    const Eigen::MatrixXd design = sampler_.generate( configuration_.number_of_points, coefficient_count );

    for ( Eigen::Index row = 0;
          row < design.rows() && candidates.size() < configuration_.number_of_points; ++row )
    {
      Eigen::VectorXi integer_direction( static_cast< Eigen::Index >( coefficient_count ) );

      for ( Eigen::Index coordinate = 0; coordinate < integer_direction.size(); ++coordinate )
      {
        // Map u in [0,1) to {-L,...,L}.
        const std::size_t level =
          std::min( 2 * radius,
                    static_cast< std::size_t >( std::floor(
                      static_cast< double >( level_count ) * design( row, coordinate ) ) ) );
        integer_direction( coordinate ) = static_cast< int >( level ) - static_cast< int >( radius );
      }

      integer_direction = projectOntoIntegerL1Ball( integer_direction, radius );
      if ( integer_direction.cwiseAbs().sum() == 0 )
      {
        continue;
      }

      if ( configuration_.remove_duplicates )
      {
        const std::vector< int > key( integer_direction.data(),
                                      integer_direction.data() + integer_direction.size() );
        if ( !used_directions.insert( key ).second )
        {
          continue;
        }
      }

      // Projection weakens the original continuous stratification, but the
      // final point remains exactly on M^k = {x^k + delta^k D y : y in Z^p}.
      candidates.push_back(
        current_candidate.parameters + mesh_size * directions * integer_direction.cast< double >() );
    }
  }

  static_cast< void >( iteration );
  return candidates;
}


void LatinHypercubeMeshGPSSearchStrategy::tell( const CandidateEvaluation &current_candidate,
                                                const std::vector< CandidateEvaluation > &evaluations,
                                                double mesh_size )
{
  static_cast< void >( current_candidate );
  static_cast< void >( evaluations );
  static_cast< void >( mesh_size );
}


void LatinHypercubeMeshGPSSearchStrategy::reset()
{
  sampler_.reset();
}


SuccessfulDirectionGPSSearchStrategy::SuccessfulDirectionGPSSearchStrategy(
  SuccessfulDirectionGPSSearchConfiguration configuration ) :
    configuration_( std::move( configuration ) ), initial_search_strategy_( configuration_.initial_search ), V_k_(),
    d_succ_(), has_successful_direction_( false )
{
}


std::vector< Eigen::VectorXd >
SuccessfulDirectionGPSSearchStrategy::ask( const CandidateEvaluation &current_candidate, double mesh_size,
                                           Eigen::Ref< const Eigen::MatrixXd > directions, std::size_t iteration )
{
  const Eigen::Index coefficient_count = directions.cols();

  // V_k = {0, e_1, ..., e_p}. Its dimension becomes known at the first ask.
  if ( V_k_.size() == 0 )
  {
    V_k_ = Eigen::MatrixXi::Zero( coefficient_count, coefficient_count + 1 );
    V_k_.rightCols( coefficient_count ).setIdentity();
  }

  if ( !has_successful_direction_ )
  {
    return initial_search_strategy_.ask( current_candidate, mesh_size, directions, iteration );
  }

  std::vector< Eigen::VectorXd > candidates;
  candidates.reserve( configuration_.multipliers.size() * static_cast< std::size_t >( V_k_.cols() ) );

  for ( const int multiplier : configuration_.multipliers )
  {
    for ( Eigen::Index column = 0; column < V_k_.cols(); ++column )
    {
      const Eigen::VectorXd generated_direction =
        static_cast< double >( multiplier ) * d_succ_ + directions * V_k_.col( column ).cast< double >();

      candidates.push_back( current_candidate.parameters + mesh_size * generated_direction );
    }
  }

  return candidates;
}


void SuccessfulDirectionGPSSearchStrategy::tell( const CandidateEvaluation &current_candidate,
                                                 const std::vector< CandidateEvaluation > &evaluations,
                                                 double mesh_size )
{
  const std::optional< CandidateEvaluation > best_candidate = extractBestCandidate( evaluations );
  if ( !best_candidate.has_value() )
  {
    return;
  }

  const double best_value = best_candidate->meanValue();
  const double current_value = current_candidate.meanValue();

  if ( best_value < current_value )
  {
    // Reconstruct the successful transformed direction before GPS changes the mesh size.
    d_succ_ = ( best_candidate->parameters - current_candidate.parameters ) / mesh_size;
    has_successful_direction_ = true;
  }
}


void SuccessfulDirectionGPSSearchStrategy::reset()
{
  V_k_.resize( 0, 0 );
  d_succ_.resize( 0 );
  has_successful_direction_ = false;
  initial_search_strategy_.reset();
}


SurrogateGPSSearchStrategy::SurrogateGPSSearchStrategy( SurrogateGPSSearchConfiguration configuration,
                                                        std::unique_ptr< SurrogateModel > surrogate_model ) :
    random_strategy_( configuration.random_points_strategy ), configuration_( std::move( configuration ) ),
    surrogate_model_( std::move( surrogate_model ) ), history_()
{
  if ( !surrogate_model_ )
  {
    throw std::invalid_argument( "SurrogateGPSSearchStrategy: surrogate model cannot be null" );
  }
  if ( !std::isfinite( configuration_.selected_fraction ) || configuration_.selected_fraction <= 0.0 ||
       configuration_.selected_fraction > 1.0 )
  {
    throw std::invalid_argument( "SurrogateGPSSearchStrategy: selected fraction must be between zero and one" );
  }
}


std::vector< Eigen::VectorXd > SurrogateGPSSearchStrategy::ask( const CandidateEvaluation &current_candidate,
                                                                double mesh_size,
                                                                Eigen::Ref< const Eigen::MatrixXd > directions,
                                                                std::size_t iteration )
{
  std::vector< Eigen::VectorXd > candidates =
    random_strategy_.ask( current_candidate, mesh_size, directions, iteration );

  if ( surrogate_model_->fit( history_, current_candidate.parameters, mesh_size ) )
  {
    std::sort( candidates.begin(), candidates.end(), [&]( const Eigen::VectorXd &left, const Eigen::VectorXd &right )
               { return surrogate_model_->predict( left ) < surrogate_model_->predict( right ); } );

    const std::size_t selected_count =
      std::min( candidates.size(),
                std::max( std::size_t{ 1 },
                          static_cast< std::size_t >( std::ceil( static_cast< double >( candidates.size() ) *
                                                                 configuration_.selected_fraction ) ) ) );
    candidates.resize( selected_count );
  }

  return candidates;
}


void SurrogateGPSSearchStrategy::tell( const CandidateEvaluation &current_candidate,
                                       const std::vector< CandidateEvaluation > &evaluations, double mesh_size )
{
  static_cast< void >( current_candidate );
  static_cast< void >( mesh_size );

  // Keep every evaluated search or poll point available to the surrogate model.
  for ( const CandidateEvaluation &candidate : evaluations )
  {
    if ( candidate.status == CandidateEvaluationStatus::Succeeded )
    {
      history_.push_back( candidate );
    }
  }
}


void SurrogateGPSSearchStrategy::reset()
{
  random_strategy_.reset();
  history_.clear();
  surrogate_model_->reset();
}
