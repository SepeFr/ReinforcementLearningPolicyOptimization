#include "optimizations_methods/gps/GPSMethod.hpp"
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include "ConfigurationEvaluationError.hpp"

namespace
{
  std::unique_ptr< SurrogateModel > createSurrogateModel( const SurrogateModelConfiguration &configuration )
  {
    return std::visit(
      []( const auto &selected_configuration ) -> std::unique_ptr< SurrogateModel >
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, LocalQuadraticSurrogateConfiguration > )
        {
          return std::make_unique< LocalQuadraticSurrogate >( selected_configuration );
        }
        else
        {
          return std::make_unique< RBFSurrogate >( selected_configuration );
        }
      },
      configuration );
  }

  std::unique_ptr< GPSSearchStrategy > createGPSSearchStrategy( const GPSSearchConfiguration &configuration )
  {
    return std::visit(
      []( const auto &selected_configuration ) -> std::unique_ptr< GPSSearchStrategy >
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, EmptyGPSSearchConfiguration > )
        {
          return std::make_unique< EmptyGPSSearchStrategy >( selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, RandomMeshGPSSearchConfiguration > )
        {
          return std::make_unique< RandomMeshGPSSearchStrategy >( selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, LatinHypercubeMeshGPSSearchConfiguration > )
        {
          return std::make_unique< LatinHypercubeMeshGPSSearchStrategy >( selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, SuccessfulDirectionGPSSearchConfiguration > )
        {
          return std::make_unique< SuccessfulDirectionGPSSearchStrategy >( selected_configuration );
        }
        else
        {
          return std::make_unique< SurrogateGPSSearchStrategy >( selected_configuration,
                                                                 createSurrogateModel( selected_configuration.model ) );
        }
      },
      configuration );
  }
} // namespace

GPSMethod::GPSMethod( InitialParametersStrategy *initial_parameters_strategy ) :
    GPSMethod( GeneralizedPatternSearchHyperparameters{}, initial_parameters_strategy )
{
}


GPSMethod::GPSMethod( const GeneralizedPatternSearchHyperparameters &hyperparameters,
                      InitialParametersStrategy *initial_parameters_strategy ) :
    OptimizerMethod( initial_parameters_strategy ), hyperparameters_( hyperparameters ),
    search_strategy_( createGPSSearchStrategy( hyperparameters.search ) ),
    poll_strategy_(
      std::make_unique< CompleteGPSPollStrategy >( std::get< CompleteGPSPollConfiguration >( hyperparameters.poll ) ) ),
    x_k_(), f_x_k_( 0.0 ), D_(), delta_k_( hyperparameters.initial_mesh_size ), k_( 0 ), phase_( Phase::Initialization )
{
  if ( !std::isfinite( hyperparameters_.initial_mesh_size ) || hyperparameters_.initial_mesh_size <= 0.0 )
  {
    throw std::invalid_argument( "GPSMethod: initial mesh size must be finite and greater than zero" );
  }
  if ( !std::isfinite( hyperparameters_.mesh_size_adjustment ) || hyperparameters_.mesh_size_adjustment <= 0.0 ||
       hyperparameters_.mesh_size_adjustment >= 1.0 )
  {
    throw std::invalid_argument( "GPSMethod: mesh size adjustment must be finite and between zero and one" );
  }
  if ( !std::isfinite( hyperparameters_.stopping_mesh_size ) || hyperparameters_.stopping_mesh_size < 0.0 )
  {
    throw std::invalid_argument( "GPSMethod: stopping mesh size must be finite and non-negative" );
  }
}


void GPSMethod::constructDMatrix()
{
  const Eigen::Index parameter_count = x_k_.size();
  if ( parameter_count == 0 )
  {
    throw std::invalid_argument( "GPSMethod: initial parameters cannot be empty" );
  }

  const Eigen::MatrixXd G =
    hyperparameters_.generating_matrix.value_or( Eigen::MatrixXd::Identity( parameter_count, parameter_count ) );

  if ( G.rows() != parameter_count || G.cols() != parameter_count )
  {
    throw std::invalid_argument( "GPSMethod: generating matrix dimensions must match the parameter count" );
  }
  if ( !G.allFinite() )
  {
    throw std::invalid_argument( "GPSMethod: generating matrix must contain only finite values" );
  }
  if ( !Eigen::FullPivLU< Eigen::MatrixXd >( G ).isInvertible() )
  {
    throw std::invalid_argument( "GPSMethod: generating matrix must be invertible" );
  }

  Eigen::MatrixXi Z;
  if ( std::holds_alternative< MinimalPositiveBasisConfiguration >( hyperparameters_.positive_basis ) )
  {
    Z.resize( parameter_count, parameter_count + 1 );
    Z.leftCols( parameter_count ).setIdentity();
    Z.col( parameter_count ).setConstant( -1 );
  }
  else if ( std::holds_alternative< SymmetricPositiveBasisConfiguration >( hyperparameters_.positive_basis ) )
  {
    Z.resize( parameter_count, 2 * parameter_count );
    Z.leftCols( parameter_count ).setIdentity();
    Z.rightCols( parameter_count ) = -Eigen::MatrixXi::Identity( parameter_count, parameter_count );
  }
  else
  {
    throw std::invalid_argument( "GPSMethod: invalid positive basis configuration" );
  }

  D_ = G * Z.cast< double >();
}


std::vector< Eigen::VectorXd > GPSMethod::ask()
{
  switch ( phase_ )
  {
    case Phase::Initialization:
    {
      x_k_ = initial_parameters_strategy_->generateInitialParameters();
      f_x_k_ = 0.0;
      delta_k_ = hyperparameters_.initial_mesh_size;
      k_ = 0;

      constructDMatrix();
      phase_ = Phase::WaitingInitialization;
      return { x_k_ };
    }
    case Phase::Search:
    {
      CandidateEvaluation current_candidate;
      current_candidate.parameters = x_k_;
      current_candidate.evaluation.appendSample( f_x_k_ );
      current_candidate.status = CandidateEvaluationStatus::Succeeded;
      std::vector< Eigen::VectorXd > candidates = search_strategy_->ask( current_candidate, delta_k_, D_, k_ );

      if ( !candidates.empty() )
      {
        phase_ = Phase::WaitingSearch;
        return candidates;
      }

      phase_ = Phase::Poll;
      [[fallthrough]];
    }
    case Phase::Poll:
    {
      const Eigen::MatrixXd directions = poll_strategy_->directions( D_, k_ );
      std::vector< Eigen::VectorXd > candidates;
      candidates.reserve( static_cast< std::size_t >( directions.cols() ) );

      for ( Eigen::Index j = 0; j < directions.cols(); ++j )
      {
        candidates.push_back( x_k_ + delta_k_ * directions.col( j ) );
      }

      phase_ = Phase::WaitingPoll;
      return candidates;
    }
    default:
      throw std::logic_error( "GPSMethod: ask called while waiting for candidate evaluations" );
  }
}

GPSMethod::Phase GPSMethod::phase() const { return phase_; }

CandidateEvaluation GPSMethod::currentCandidate() const
{
  CandidateEvaluation current_candidate;
  current_candidate.parameters = x_k_;
  current_candidate.evaluation.appendSample( f_x_k_ );
  current_candidate.status = CandidateEvaluationStatus::Succeeded;
  return current_candidate;
}

double GPSMethod::meshSize() const { return delta_k_; }

void GPSMethod::completeSearchStep(
  const CandidateEvaluation &incumbent,
  const std::vector< CandidateEvaluation > &trial_evaluations,
  const std::optional< CandidateEvaluation > &accepted_candidate )
{
  // Strategies receive the complete feedback before the incumbent or mesh changes.
  search_strategy_->tell( incumbent, trial_evaluations, delta_k_ );
  f_x_k_ = incumbent.meanValue();

  if ( accepted_candidate.has_value() )
  {
    x_k_ = accepted_candidate->parameters;
    f_x_k_ = accepted_candidate->meanValue();
    delta_k_ /= hyperparameters_.mesh_size_adjustment;
    ++k_;
    phase_ = Phase::Search;
    return;
  }

  phase_ = Phase::Poll;
}

void GPSMethod::completePollStep(
  const CandidateEvaluation &incumbent,
  const std::vector< CandidateEvaluation > &trial_evaluations,
  const std::optional< CandidateEvaluation > &accepted_candidate )
{
  // Poll observations also enrich stateful search strategies.
  search_strategy_->tell( incumbent, trial_evaluations, delta_k_ );
  f_x_k_ = incumbent.meanValue();

  if ( accepted_candidate.has_value() )
  {
    x_k_ = accepted_candidate->parameters;
    f_x_k_ = accepted_candidate->meanValue();
    delta_k_ /= hyperparameters_.mesh_size_adjustment;
  }
  else
  {
    delta_k_ *= hyperparameters_.mesh_size_adjustment;
  }

  ++k_;
  phase_ = Phase::Search;
}


OptimizerMethodUpdate GPSMethod::tellImpl( const std::vector< CandidateEvaluation > &evaluations )
{
  if ( evaluations.empty() )
  {
    throw std::invalid_argument( "GPSMethod: evaluations cannot be empty" );
  }

  switch ( phase_ )
  {
    case Phase::WaitingInitialization:
    {
      if ( evaluations.front().status != CandidateEvaluationStatus::Succeeded )
      {
        throw ConfigurationEvaluationError( "GPSMethod: initial candidate evaluation failed" );
      }

      x_k_ = evaluations.front().parameters;
      f_x_k_ = evaluations.front().meanValue();

      // Make the evaluated initial point available to strategies that maintain a history.
      search_strategy_->tell( evaluations.front(), evaluations, delta_k_ );

      phase_ = Phase::Search;
      return OptimizerMethodUpdate{ false };
    }
    case Phase::WaitingSearch:
    {
      const std::optional< CandidateEvaluation > batch_best_candidate = extractBestCandidate( evaluations );
      const std::optional< CandidateEvaluation > accepted_candidate =
        batch_best_candidate.has_value() && batch_best_candidate->meanValue() < f_x_k_
          ? batch_best_candidate
          : std::nullopt;

      completeSearchStep( currentCandidate(), evaluations, accepted_candidate );
      return OptimizerMethodUpdate{ accepted_candidate.has_value() };
    }
    case Phase::WaitingPoll:
    {
      const std::optional< CandidateEvaluation > batch_best_candidate = extractBestCandidate( evaluations );
      const std::optional< CandidateEvaluation > accepted_candidate =
        batch_best_candidate.has_value() && batch_best_candidate->meanValue() < f_x_k_
          ? batch_best_candidate
          : std::nullopt;

      completePollStep( currentCandidate(), evaluations, accepted_candidate );
      return {};
    }
    default:
      throw std::logic_error( "GPSMethod: tell called without pending candidate evaluations" );
  }
}


OptimizerMethod::ConvergenceStatus GPSMethod::convergenceStatus() const
{
  if ( hyperparameters_.stopping_mesh_size == 0.0 )
  {
    return ConvergenceStatus::NoInternalCriterion;
  }

  if ( k_ == 0 )
  {
    return ConvergenceStatus::InProgress;
  }

  return delta_k_ < hyperparameters_.stopping_mesh_size ? ConvergenceStatus::Converged : ConvergenceStatus::InProgress;
}


void GPSMethod::reset()
{
  OptimizerMethod::reset();
  x_k_.resize( 0 );
  f_x_k_ = 0.0;
  D_.resize( 0, 0 );
  delta_k_ = hyperparameters_.initial_mesh_size;
  k_ = 0;
  phase_ = Phase::Initialization;

  search_strategy_->reset();
  poll_strategy_->reset();
}
