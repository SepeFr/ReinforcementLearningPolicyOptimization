#include "optimizations_methods/nelder_mead/NelderMeadMethod.hpp"
#include "LatinHypercubeSampler.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include "ConfigurationEvaluationError.hpp"
#include "optimizations_methods/nelder_mead/NelderMeadStoppingCriterion.hpp"

namespace
{
  CandidateEvaluation unevaluatedCandidate( Eigen::VectorXd parameters )
  {
    return CandidateEvaluation{
      std::move( parameters ), ObjectiveEvaluation{}, CandidateEvaluationStatus::NotEvaluated
    };
  }

  bool allEvaluationsSucceeded( const std::vector< CandidateEvaluation > &evaluations )
  {
    return std::all_of( evaluations.begin(), evaluations.end(), []( const CandidateEvaluation &candidate )
                        { return candidate.status == CandidateEvaluationStatus::Succeeded; } );
  }

  std::unique_ptr< NelderMeadStoppingStrategy >
  createNelderMeadStoppingStrategy( const NelderMeadStoppingConfiguration &configuration )
  {
    return std::visit(
      []( const auto &selected_configuration ) -> std::unique_ptr< NelderMeadStoppingStrategy >
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, DispersionStoppingConfiguration > )
        {
          return std::make_unique< DispersionStoppingCriterion >( selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, RelativeVariationStoppingConfiguration > )
        {
          return std::make_unique< RelativeVariationStoppingCriterion >( selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, StandardDeviationStoppingConfiguration > )
        {
          return std::make_unique< StandardDeviationStoppingCriterion >( selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, HeightDifferenceStoppingConfiguration > )
        {
          return std::make_unique< HeightDifferenceStoppingCriterion >( selected_configuration );
        }
        else
        {
          return std::make_unique< RelativeFunctionToleranceStoppingCriterion >( selected_configuration );
        }
      },
      configuration );
  }
} // namespace

NelderMeadMethod::NelderMeadMethod( std::size_t number_of_parameters,
                                    InitialParametersStrategy *initial_parameters_strategy ) :
    NelderMeadMethod( NelderMeadHyperparameters{}, number_of_parameters, initial_parameters_strategy )
{
}


NelderMeadMethod::NelderMeadMethod( const NelderMeadHyperparameters &hyperparameters, std::size_t number_of_parameters,
                                    InitialParametersStrategy *initial_parameters_strategy ) :
    OptimizerMethod( initial_parameters_strategy ), centroid_( static_cast< Eigen::Index >( number_of_parameters ) ),
    number_of_vertices_( number_of_parameters + 1 ), hyperparameters_( hyperparameters ),
    stopping_strategy_( createNelderMeadStoppingStrategy( hyperparameters.stopping_criterion ) )
{
  // Intentional extension of Algorithm 5.1: the coefficients are configurable.
  // Their default values reproduce the algorithm, while different values permit common Nelder-Mead variants.
  if ( hyperparameters_.initial_simplex_scale <= 0.0 )
  {
    throw std::invalid_argument( "NelderMeadMethod: initial simplex scale must be positive" );
  }

  initializeSimplex();
}

void NelderMeadMethod::initializeSimplex()
{
  simplex_.clear();
  simplex_.reserve( number_of_vertices_ );

  std::visit(
    [&]( const auto &configuration )
    {
      using ConfigurationType = std::decay_t< decltype( configuration ) >;

      if constexpr ( std::is_same_v< ConfigurationType, ClassicalLocalSimplexConfiguration > )
      {
        initializeClassicalLocalSimplex();
      }
      else if constexpr ( std::is_same_v< ConfigurationType, LatinHypercubeLocalSimplexConfiguration > )
      {
        initializeLatinHypercubeLocalSimplex( configuration );
      }
      else if constexpr ( std::is_same_v< ConfigurationType, LatinHypercubeGlobalSimplexConfiguration > )
      {
        initializeLatinHypercubeGlobalSimplex( configuration );
      }
      else
      {
        initializeProvidedSimplex();
      }
    },
    hyperparameters_.initialization );

  initial_simplex_ = simplex_;
}


void NelderMeadMethod::initializeClassicalLocalSimplex()
{
  const Eigen::VectorXd center = initial_parameters_strategy_->generateInitialParameters();
  const Eigen::Index dimension = static_cast< Eigen::Index >( number_of_vertices_ - 1 );
  if ( center.size() != dimension )
  {
    throw std::invalid_argument( "NelderMeadMethod: initial point dimension does not match the problem" );
  }

  simplex_.push_back( unevaluatedCandidate( center ) );

  // x_i = x_0 + s e_i. The simplex geometry is independent of how x_0
  // was initialized, so zero-initialized bias coordinates are still spanned.
  for ( Eigen::Index coordinate = 0; coordinate < dimension; ++coordinate )
  {
    Eigen::VectorXd vertex = center;
    vertex( coordinate ) += hyperparameters_.initial_simplex_scale;
    simplex_.push_back( unevaluatedCandidate( std::move( vertex ) ) );
  }
}


void NelderMeadMethod::initializeLatinHypercubeLocalSimplex(
  const LatinHypercubeLocalSimplexConfiguration &configuration )
{
  if ( configuration.maximum_attempts == 0 )
  {
    throw std::invalid_argument( "NelderMeadMethod: LHS maximum attempts must be greater than zero" );
  }

  const Eigen::VectorXd center = initial_parameters_strategy_->generateInitialParameters();
  const std::size_t dimension = number_of_vertices_ - 1;
  if ( center.size() != static_cast< Eigen::Index >( dimension ) )
  {
    throw std::invalid_argument( "NelderMeadMethod: initial point dimension does not match the problem" );
  }

  LatinHypercubeSampler sampler( configuration.random_seed );

  for ( std::size_t attempt = 0; attempt < configuration.maximum_attempts; ++attempt )
  {
    const Eigen::MatrixXd design = sampler.generate( dimension, dimension );
    simplex_.clear();
    simplex_.push_back( unevaluatedCandidate( center ) );

    for ( Eigen::Index row = 0; row < design.rows(); ++row )
    {
      // x_i = x_0 + s(2u_i - 1), keeping the LHS simplex local to x_0.
      const Eigen::VectorXd centered_direction = ( 2.0 * design.row( row ).array() - 1.0 ).matrix().transpose();
      simplex_.push_back(
        unevaluatedCandidate( center + hyperparameters_.initial_simplex_scale * centered_direction ) );
    }

    if ( hasFullAffineRank() )
    {
      return;
    }
  }

  throw ConfigurationEvaluationError( "NelderMeadMethod: unable to generate a full-rank local LHS simplex" );
}


void NelderMeadMethod::initializeLatinHypercubeGlobalSimplex(
  const LatinHypercubeGlobalSimplexConfiguration &configuration )
{
  const std::size_t dimension = number_of_vertices_ - 1;
  const Eigen::Index eigen_dimension = static_cast< Eigen::Index >( dimension );

  if ( configuration.maximum_attempts == 0 )
  {
    throw std::invalid_argument( "NelderMeadMethod: LHS maximum attempts must be greater than zero" );
  }
  if ( configuration.lower_bound.size() != eigen_dimension || configuration.upper_bound.size() != eigen_dimension )
  {
    throw std::invalid_argument( "NelderMeadMethod: global LHS bounds must match the problem dimension" );
  }
  if ( !( configuration.lower_bound.array() <= configuration.upper_bound.array() ).all() )
  {
    throw std::invalid_argument( "NelderMeadMethod: global LHS lower bounds cannot exceed upper bounds" );
  }

  LatinHypercubeSampler sampler( configuration.random_seed );
  const Eigen::VectorXd interval = configuration.upper_bound - configuration.lower_bound;

  for ( std::size_t attempt = 0; attempt < configuration.maximum_attempts; ++attempt )
  {
    const Eigen::MatrixXd design = sampler.generate( number_of_vertices_, dimension );
    simplex_.clear();

    for ( Eigen::Index row = 0; row < design.rows(); ++row )
    {
      // x_i = l + u_i odot (h - l). These bounds define the initial global
      // Latin-hypercube design interval.
      const Eigen::VectorXd vertex =
        configuration.lower_bound + ( design.row( row ).transpose().array() * interval.array() ).matrix();
      simplex_.push_back( unevaluatedCandidate( vertex ) );
    }

    if ( hasFullAffineRank() )
    {
      return;
    }
  }

  throw ConfigurationEvaluationError( "NelderMeadMethod: unable to generate a full-rank global LHS simplex" );
}


void NelderMeadMethod::initializeProvidedSimplex()
{
  for ( std::size_t index = 0; index < number_of_vertices_; ++index )
  {
    simplex_.push_back( unevaluatedCandidate( initial_parameters_strategy_->generateInitialParameters() ) );
  }

  if ( !hasFullAffineRank() )
  {
    throw std::invalid_argument( "NelderMeadMethod: provided simplex must have full affine rank" );
  }
}


bool NelderMeadMethod::hasFullAffineRank() const
{
  const Eigen::Index dimension = static_cast< Eigen::Index >( number_of_vertices_ - 1 );
  if ( simplex_.size() != number_of_vertices_ || simplex_.front().parameters.size() != dimension )
  {
    return false;
  }

  Eigen::MatrixXd difference_matrix( dimension, dimension );
  for ( Eigen::Index column = 0; column < dimension; ++column )
  {
    const Eigen::VectorXd &vertex = simplex_.at( static_cast< std::size_t >( column + 1 ) ).parameters;
    if ( vertex.size() != dimension )
    {
      return false;
    }
    difference_matrix.col( column ) = vertex - simplex_.front().parameters;
  }

  Eigen::FullPivLU< Eigen::MatrixXd > decomposition( difference_matrix );
  decomposition.setThreshold( std::sqrt( std::numeric_limits< double >::epsilon() ) );
  return decomposition.rank() == dimension;
}

std::vector< Eigen::VectorXd > NelderMeadMethod::simplexParameters() const
{
  std::vector< Eigen::VectorXd > parameters;
  parameters.reserve( simplex_.size() );

  for ( const CandidateEvaluation &vertex : simplex_ )
  {
    parameters.push_back( vertex.parameters );
  }

  return parameters;
}


std::vector< Eigen::VectorXd > NelderMeadMethod::ask()
{
  const std::size_t worst_index = number_of_vertices_ - 1;

  switch ( phase_ )
  {
    case Phase::InitialSimplex:
      // The complete simplex is evaluated only during initialization and after reset().
      phase_ = Phase::WaitingInitialSimplex;
      return simplexParameters();
    case Phase::Reflection:
      reflected_point_ =
        centroid_ +
        hyperparameters_.reflection_coefficient * ( centroid_ - simplex_.at( worst_index ).parameters );
      phase_ = Phase::WaitingReflection;
      return { reflected_point_ };
    case Phase::Expansion:
      expansion_point_ =
        centroid_ +
        hyperparameters_.expansion_coefficient * ( centroid_ - simplex_.at( worst_index ).parameters );
      phase_ = Phase::WaitingExpansion;
      return { expansion_point_ };
    case Phase::OutsideContraction:
      outside_contraction_point_ =
        centroid_ +
        hyperparameters_.outside_contraction_coefficient * ( centroid_ - simplex_.at( worst_index ).parameters );
      phase_ = Phase::WaitingOutsideContraction;
      return { outside_contraction_point_ };
    case Phase::InsideContraction:
      inside_contraction_point_ =
        centroid_ +
        hyperparameters_.inside_contraction_coefficient * ( centroid_ - simplex_.at( worst_index ).parameters );
      phase_ = Phase::WaitingInsideContraction;
      return { inside_contraction_point_ };
    case Phase::Shrink:
    {
      // Construct the shrink only when ask enters the operation phase.
      std::vector< Eigen::VectorXd > shrunk_parameters;
      shrunk_parameters.reserve( number_of_vertices_ - 1 );
      for ( std::size_t index = 1; index < number_of_vertices_; index++ )
      {
        const Eigen::VectorXd parameters =
          simplex_.front().parameters +
          hyperparameters_.shrink_coefficient *
            ( simplex_.at( index ).parameters - simplex_.front().parameters );
        simplex_.at( index ) = unevaluatedCandidate( parameters );
        shrunk_parameters.push_back( parameters );
      }

      // The best vertex is unchanged by shrink, so its stored evaluation remains valid.
      phase_ = Phase::WaitingShrink;
      return shrunk_parameters;
    }
    default:
      throw std::logic_error( "NelderMeadMethod: ask called while waiting for candidate evaluations" );
  }
}

NelderMeadMethod::Phase NelderMeadMethod::phase() const { return phase_; }

void NelderMeadMethod::setPhase( Phase phase ) { phase_ = phase; }

std::size_t NelderMeadMethod::vertexCount() const { return number_of_vertices_; }

std::size_t NelderMeadMethod::iterationCount() const { return iteration_counter_; }

const std::vector< CandidateEvaluation > &NelderMeadMethod::simplex() const { return simplex_; }

CandidateEvaluation &NelderMeadMethod::simplexVertex( std::size_t index ) { return simplex_.at( index ); }

const Eigen::VectorXd &NelderMeadMethod::reflectedPoint() const { return reflected_point_; }

const CandidateEvaluation &NelderMeadMethod::reflectedCandidate() const
{
  return reflected_candidate_;
}

void NelderMeadMethod::orderSimplex()
{
  std::sort( simplex_.begin(), simplex_.end(),
             []( const CandidateEvaluation &left, const CandidateEvaluation &right )
             { return left.meanValue() < right.meanValue(); } );

  f_.clear();
  f_.reserve( simplex_.size() );
  for ( const CandidateEvaluation &vertex : simplex_ )
  {
    f_.push_back( vertex.meanValue() );
  }
  f_best_ = f_.front();
}

void NelderMeadMethod::prepareIteration()
{
  orderSimplex();

  // Convergence is checked only with a complete, consistently ordered simplex.
  has_converged_ = stopping_strategy_->shouldStop( simplexParameters(), f_ );
  if ( has_converged_ )
  {
    return;
  }

  centroid_.setZero();
  for ( std::size_t index = 0; index + 1 < number_of_vertices_; ++index )
  {
    centroid_ += simplex_.at( index ).parameters;
  }
  centroid_ /= static_cast< double >( number_of_vertices_ - 1 );
  phase_ = Phase::Reflection;
}

void NelderMeadMethod::prepareNextIteration() { prepareIteration(); }

void NelderMeadMethod::processReflectionEvaluation( const CandidateEvaluation &evaluation )
{
  const std::size_t worst_index = number_of_vertices_ - 1;
  const std::size_t second_worst_index = number_of_vertices_ - 2;

  reflected_candidate_ = evaluation;
  if ( evaluation.status != CandidateEvaluationStatus::Succeeded )
  {
    phase_ = Phase::InsideContraction;
    return;
  }

  f_r_ = evaluation.meanValue();

  if ( f_best_ <= f_r_ && f_r_ < f_.at( second_worst_index ) )
  {
    completeIteration( evaluation );
  }
  else if ( f_r_ < f_best_ )
  {
    phase_ = Phase::Expansion;
  }
  else if ( f_.at( second_worst_index ) <= f_r_ && f_r_ < f_.at( worst_index ) )
  {
    phase_ = Phase::OutsideContraction;
  }
  else
  {
    phase_ = Phase::InsideContraction;
  }
}

void NelderMeadMethod::completeIteration( const CandidateEvaluation &replacement )
{
  simplex_.back() = replacement;
  ++iteration_counter_;
  prepareNextIteration();
}

void NelderMeadMethod::completeIterationWithoutReplacement()
{
  ++iteration_counter_;
  prepareNextIteration();
}


OptimizerMethodUpdate NelderMeadMethod::tellImpl( const std::vector< CandidateEvaluation > &evaluations )
{
  const std::size_t worst_index = number_of_vertices_ - 1;

  switch ( phase_ )
  {
    case Phase::WaitingInitialSimplex:
    {
      if ( evaluations.size() != number_of_vertices_ )
      {
        throw std::invalid_argument( "NelderMeadMethod: simplex evaluation count does not match vertex count" );
      }
      if ( !allEvaluationsSucceeded( evaluations ) )
      {
        throw ConfigurationEvaluationError(
          "NelderMeadMethod: every initial simplex vertex evaluation must succeed" );
      }

      simplex_ = evaluations;
      prepareIteration();
      return OptimizerMethodUpdate{ false };
    }

    case Phase::WaitingShrink:
    {
      if ( evaluations.size() != number_of_vertices_ - 1 )
      {
        throw std::invalid_argument(
          "NelderMeadMethod: shrink evaluation count does not match changed vertex count" );
      }
      if ( !allEvaluationsSucceeded( evaluations ) )
      {
        throw ConfigurationEvaluationError(
          "NelderMeadMethod: every changed shrink vertex evaluation must succeed" );
      }

      for ( std::size_t index = 0; index < evaluations.size(); ++index )
      {
        simplex_.at( index + 1 ) = evaluations.at( index );
      }

      ++iteration_counter_;
      prepareNextIteration();
      return {};
    }

    case Phase::WaitingReflection:
    {
      if ( evaluations.size() != 1 )
      {
        throw std::invalid_argument( "NelderMeadMethod: reflection requires exactly one evaluation" );
      }

      const std::size_t previous_iteration_count = iteration_counter_;
      processReflectionEvaluation( evaluations.front() );
      return OptimizerMethodUpdate{ iteration_counter_ > previous_iteration_count };
    }

    case Phase::WaitingExpansion:
    {
      if ( evaluations.size() != 1 )
      {
        throw std::invalid_argument( "NelderMeadMethod: expansion requires exactly one evaluation" );
      }

      if ( evaluations.front().status != CandidateEvaluationStatus::Succeeded )
      {
        // Reflection is already valid and remains preferable to a failed expansion.
        completeIteration( reflected_candidate_ );
        return {};
      }

      const double f_e = evaluations.front().meanValue();

      // Keep the better point between expansion and reflection.
      completeIteration( f_e < f_r_ ? evaluations.front() : reflected_candidate_ );
      return {};
    }

    case Phase::WaitingOutsideContraction:
    {
      if ( evaluations.size() != 1 )
      {
        throw std::invalid_argument( "NelderMeadMethod: outside contraction requires exactly one evaluation" );
      }

      if ( evaluations.front().status != CandidateEvaluationStatus::Succeeded )
      {
        completeIteration( reflected_candidate_ );
        return {};
      }

      const double f_oc = evaluations.front().meanValue();

      completeIteration( f_oc < f_r_ ? evaluations.front() : reflected_candidate_ );
      return {};
    }

    case Phase::WaitingInsideContraction:
    {
      if ( evaluations.size() != 1 )
      {
        throw std::invalid_argument( "NelderMeadMethod: inside contraction requires exactly one evaluation" );
      }

      const std::size_t previous_iteration_count = iteration_counter_;
      const bool contraction_succeeded = evaluations.front().status == CandidateEvaluationStatus::Succeeded;
      if ( contraction_succeeded && evaluations.front().meanValue() < f_.at( worst_index ) )
      {
        // Accept the inside contraction when it improves the worst vertex.
        completeIteration( evaluations.front() );
      }
      else
      {
        // The next ask constructs the shrink before requesting its evaluations.
        phase_ = Phase::Shrink;
      }
      return OptimizerMethodUpdate{ iteration_counter_ > previous_iteration_count };
    }
    default:
      throw std::logic_error( "NelderMeadMethod: tell called without pending candidate evaluations" );
  }
}


OptimizerMethod::ConvergenceStatus NelderMeadMethod::convergenceStatus() const
{
  return has_converged_ ? ConvergenceStatus::Converged : ConvergenceStatus::InProgress;
}


void NelderMeadMethod::reset()
{
  OptimizerMethod::reset();
  phase_ = Phase::InitialSimplex;
  simplex_ = initial_simplex_;
  reflected_candidate_ = CandidateEvaluation{};
  f_.clear();
  centroid_.setZero();
  f_best_ = 0.0;
  f_r_ = 0.0;
  iteration_counter_ = 0;
  has_converged_ = false;
  stopping_strategy_ = createNelderMeadStoppingStrategy( hyperparameters_.stopping_criterion );
}
