#include "optimizations_methods/gps/RinottRankingSelectionGPSMethod.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include "ConfigurationEvaluationError.hpp"

namespace
{
  /** @brief Nodes and normalized probability weights for gamma-density quadrature. */
  struct GammaQuadratureRule
  {
    Eigen::VectorXd nodes; ///< Generalized Gauss--Laguerre nodes.
    Eigen::VectorXd normalized_weights; ///< Weights normalized for the target gamma density.
  };

  /**
   * @brief Builds a fixed order-32 quadrature for a chi-square distribution.
   * @param[in] degrees_of_freedom Positive chi-square degrees of freedom.
   * @return Gamma-density nodes and normalized weights produced by a
   * Golub--Welsch eigenvalue decomposition.
   * @throws ConfigurationEvaluationError If the symmetric eigensolver fails.
   */
  GammaQuadratureRule makeGammaQuadratureRule( std::size_t degrees_of_freedom )
  {
    // A chi-square variable with nu degrees of freedom is twice a
    // Gamma(nu / 2, 1) variable. Golub-Welsch constructs a generalized
    // Gauss-Laguerre rule for that gamma density.
    constexpr Eigen::Index quadrature_order = 32;
    const double alpha =
      static_cast< double >( degrees_of_freedom ) / 2.0 - 1.0;

    Eigen::MatrixXd jacobi_matrix =
      Eigen::MatrixXd::Zero( quadrature_order, quadrature_order );

    for ( Eigen::Index index = 0; index < quadrature_order; ++index )
    {
      jacobi_matrix( index, index ) =
        2.0 * static_cast< double >( index ) + alpha + 1.0;
    }

    for ( Eigen::Index index = 0; index + 1 < quadrature_order; ++index )
    {
      const double off_diagonal =
        std::sqrt(
          static_cast< double >( index + 1 ) *
          ( static_cast< double >( index + 1 ) + alpha ) );
      jacobi_matrix( index, index + 1 ) = off_diagonal;
      jacobi_matrix( index + 1, index ) = off_diagonal;
    }

    Eigen::SelfAdjointEigenSolver< Eigen::MatrixXd > decomposition(
      jacobi_matrix );
    if ( decomposition.info() != Eigen::Success )
    {
      throw ConfigurationEvaluationError(
        "RinottRankingSelectionGPSMethod: failed to construct numerical quadrature" );
    }

    GammaQuadratureRule rule;
    rule.nodes = decomposition.eigenvalues();
    rule.normalized_weights =
      decomposition.eigenvectors().row( 0 ).array().square().matrix().transpose();
    return rule;
  }

  /** @brief Evaluates the standard-normal CDF. @param[in] value Abscissa. @return \f$\Phi(value)\f$. */
  double standardNormalCdf( double value )
  {
    return 0.5 * std::erfc( -value / std::sqrt( 2.0 ) );
  }

  /**
   * @brief Numerically evaluates the probability defining the Rinott constant.
   * @param[in] constant Trial Rinott constant.
   * @param[in] candidate_count Number of eligible alternatives.
   * @param[in] degrees_of_freedom Stage-one degrees of freedom.
   * @param[in] rule Quadrature shared by both chi-square integrations.
   * @return Approximated selection probability, monotone in @p constant.
   */
  double rinottSelectionProbability(
    double constant, std::size_t candidate_count,
    std::size_t degrees_of_freedom, const GammaQuadratureRule &rule )
  {
    const double degrees = static_cast< double >( degrees_of_freedom );
    double probability = 0.0;

    for ( Eigen::Index y_index = 0; y_index < rule.nodes.size(); ++y_index )
    {
      const double y = 2.0 * rule.nodes( y_index );
      double inner_integral = 0.0;

      for ( Eigen::Index x_index = 0; x_index < rule.nodes.size(); ++x_index )
      {
        const double x = 2.0 * rule.nodes( x_index );
        const double denominator =
          std::sqrt( degrees * ( 1.0 / x + 1.0 / y ) );

        inner_integral +=
          rule.normalized_weights( x_index ) *
          standardNormalCdf( constant / denominator );
      }

      probability +=
        rule.normalized_weights( y_index ) *
        std::pow(
          inner_integral,
          static_cast< double >( candidate_count - 1 ) );
    }

    return probability;
  }

  /**
   * @brief Solves numerically for the smallest constant reaching a confidence.
   *
   * A power-of-two expansion brackets the solution in at most 64 attempts,
   * followed by 64 bisection iterations.
   *
   * @param[in] candidate_count Eligible alternatives.
   * @param[in] degrees_of_freedom Stage-one sample count minus one.
   * @param[in] confidence Required probability \f$1-\alpha_r\f$.
   * @return Rinott constant; zero for fewer than two alternatives or when zero
   * already attains the target probability.
   * @throws ConfigurationEvaluationError If quadrature fails or a finite upper bracket cannot be found.
   */
  double calculateRinottConstant(
    std::size_t candidate_count, std::size_t degrees_of_freedom,
    double confidence )
  {
    if ( candidate_count < 2 )
    {
      return 0.0;
    }

    const GammaQuadratureRule rule =
      makeGammaQuadratureRule( degrees_of_freedom );

    // The integral is monotone in g. If g = 0 already reaches the requested
    // confidence, the first-stage samples are sufficient.
    if ( rinottSelectionProbability(
           0.0, candidate_count, degrees_of_freedom, rule ) >= confidence )
    {
      return 0.0;
    }

    double lower_bound = 0.0;
    double upper_bound = 1.0;

    for ( std::size_t attempt = 0;
          attempt < 64 &&
          rinottSelectionProbability(
            upper_bound, candidate_count, degrees_of_freedom, rule ) < confidence;
          ++attempt )
    {
      upper_bound *= 2.0;
    }

    if ( !std::isfinite( upper_bound ) ||
         rinottSelectionProbability(
           upper_bound, candidate_count, degrees_of_freedom, rule ) < confidence )
    {
      throw ConfigurationEvaluationError(
        "RinottRankingSelectionGPSMethod: could not bracket the Rinott constant" );
    }

    // Bisection is sufficient because the defining probability is monotone.
    for ( std::size_t iteration = 0; iteration < 64; ++iteration )
    {
      const double midpoint = ( lower_bound + upper_bound ) / 2.0;
      if ( rinottSelectionProbability(
             midpoint, candidate_count, degrees_of_freedom, rule ) < confidence )
      {
        lower_bound = midpoint;
      }
      else
      {
        upper_bound = midpoint;
      }
    }

    return ( lower_bound + upper_bound ) / 2.0;
  }

  /**
   * @brief Creates the stochastic stopping strategy selected by a variant.
   * @param[in] configuration Stochastic criterion settings.
   * @param[in] mesh_size_tolerance GPS mesh threshold used by the marker alternative.
   * @return Unique ownership of the selected strategy.
   */
  std::unique_ptr< StochasticGPSStoppingStrategy >
  createStochasticGPSStoppingStrategy(
    const StochasticGPSStoppingConfiguration &configuration,
    double mesh_size_tolerance )
  {
    return std::visit(
      [&]( const auto &selected_configuration )
        -> std::unique_ptr< StochasticGPSStoppingStrategy >
      {
        using ConfigurationType =
          std::decay_t< decltype( selected_configuration ) >;

        if constexpr (
          std::is_same_v<
            ConfigurationType,
            NoStochasticGPSStoppingConfiguration > )
        {
          return std::make_unique< NoStochasticGPSStoppingCriterion >();
        }
        else if constexpr (
          std::is_same_v<
            ConfigurationType,
            StochasticGPSMeshSizeStoppingConfiguration > )
        {
          return std::make_unique<
            StochasticGPSMeshSizeStoppingCriterion >(
              mesh_size_tolerance );
        }
        else if constexpr (
          std::is_same_v<
            ConfigurationType,
            StochasticGPSSignificanceStoppingConfiguration > )
        {
          return std::make_unique<
            StochasticGPSSignificanceStoppingCriterion >(
              selected_configuration );
        }
        else
        {
          return std::make_unique<
            StochasticGPSNoiseToIndifferenceStoppingCriterion >(
              selected_configuration );
        }
      },
      configuration );
  }
} // namespace

RinottRankingSelectionGPSMethod::RinottRankingSelectionGPSMethod(
  InitialParametersStrategy *initial_parameters_strategy ) :
    RinottRankingSelectionGPSMethod(
      RinottRankingSelectionGPSHyperparameters{},
      initial_parameters_strategy )
{
}

RinottRankingSelectionGPSMethod::RinottRankingSelectionGPSMethod(
  const RinottRankingSelectionGPSHyperparameters &hyperparameters,
  InitialParametersStrategy *initial_parameters_strategy ) :
    GPSMethod( hyperparameters, initial_parameters_strategy ),
    hyperparameters_( hyperparameters ),
    stopping_strategy_(
      createStochasticGPSStoppingStrategy(
        hyperparameters.stopping_criterion,
        hyperparameters.stopping_mesh_size ) ),
    significance_( hyperparameters.initial_significance ),
    indifference_( hyperparameters.initial_indifference )
{
  if ( hyperparameters_.initial_sample_size < 2 )
  {
    throw std::invalid_argument(
      "RinottRankingSelectionGPSMethod: initial sample size must be at least two" );
  }
  if ( !std::isfinite( hyperparameters_.initial_significance ) ||
       hyperparameters_.initial_significance <= 0.0 ||
       hyperparameters_.initial_significance >= 1.0 )
  {
    throw std::invalid_argument(
      "RinottRankingSelectionGPSMethod: initial significance must be in (0, 1)" );
  }
  if ( !std::isfinite( hyperparameters_.initial_indifference ) ||
       hyperparameters_.initial_indifference <= 0.0 )
  {
    throw std::invalid_argument(
      "RinottRankingSelectionGPSMethod: initial indifference must be positive" );
  }
  if ( !std::isfinite( hyperparameters_.significance_decay ) ||
       hyperparameters_.significance_decay <= 0.0 ||
       hyperparameters_.significance_decay >= 1.0 )
  {
    throw std::invalid_argument(
      "RinottRankingSelectionGPSMethod: significance decay must be in (0, 1)" );
  }
  if ( !std::isfinite( hyperparameters_.indifference_decay ) ||
       hyperparameters_.indifference_decay <= 0.0 ||
       hyperparameters_.indifference_decay >= 1.0 )
  {
    throw std::invalid_argument(
      "RinottRankingSelectionGPSMethod: indifference decay must be in (0, 1)" );
  }

  std::visit(
    [&]( const auto &configuration )
    {
      using ConfigurationType = std::decay_t< decltype( configuration ) >;

      if constexpr (
        std::is_same_v<
          ConfigurationType,
          StochasticGPSMeshSizeStoppingConfiguration > )
      {
        if ( hyperparameters_.stopping_mesh_size <= 0.0 )
        {
          throw std::invalid_argument(
            "RinottRankingSelectionGPSMethod: mesh stopping tolerance must be positive" );
        }
      }
      else if constexpr (
        std::is_same_v<
          ConfigurationType,
          StochasticGPSSignificanceStoppingConfiguration > )
      {
        if ( !std::isfinite( configuration.tolerance ) ||
             configuration.tolerance <= 0.0 ||
             configuration.tolerance >= 1.0 )
        {
          throw std::invalid_argument(
            "RinottRankingSelectionGPSMethod: significance stopping tolerance must be in (0, 1)" );
        }
      }
      else if constexpr (
        std::is_same_v<
          ConfigurationType,
          StochasticGPSNoiseToIndifferenceStoppingConfiguration > )
      {
        if ( !std::isfinite( configuration.threshold ) ||
             configuration.threshold <= 0.0 )
        {
          throw std::invalid_argument(
            "RinottRankingSelectionGPSMethod: noise-to-indifference stopping threshold must be positive" );
        }
      }
    },
    hyperparameters_.stopping_criterion );
}

bool RinottRankingSelectionGPSMethod::samePoint(
  Eigen::Ref< const Eigen::VectorXd > left,
  Eigen::Ref< const Eigen::VectorXd > right )
{
  return left.size() == right.size() &&
    ( left.array() == right.array() ).all();
}

std::vector< Eigen::VectorXd >
RinottRankingSelectionGPSMethod::beginRankingSelection(
  const std::vector< Eigen::VectorXd > &trial_points )
{
  candidates_.clear();

  const Eigen::VectorXd incumbent_parameters = currentCandidate().parameters;
  for ( const Eigen::VectorXd &parameters : trial_points )
  {
    const bool already_present =
      samePoint( parameters, incumbent_parameters ) ||
      std::any_of(
        candidates_.begin(), candidates_.end(),
        [&]( const RankingSelectionCandidate &candidate )
        { return samePoint( parameters, candidate.aggregate.parameters ); } );

    if ( already_present )
    {
      continue;
    }

    CandidateEvaluation aggregate;
    aggregate.parameters = parameters;
    aggregate.status = CandidateEvaluationStatus::NotEvaluated;
    candidates_.push_back(
      RankingSelectionCandidate{
        std::move( aggregate ),
        hyperparameters_.initial_sample_size,
        true } );
  }

  incumbent_index_ = candidates_.size();
  CandidateEvaluation incumbent;
  incumbent.parameters = incumbent_parameters;
  incumbent.status = CandidateEvaluationStatus::NotEvaluated;
  candidates_.push_back(
    RankingSelectionCandidate{
      std::move( incumbent ),
      hyperparameters_.initial_sample_size,
      true } );

  ranking_selection_phase_ = RankingSelectionPhase::WaitingFirstStage;
  return buildPendingRequests();
}

std::vector< Eigen::VectorXd >
RinottRankingSelectionGPSMethod::buildPendingRequests()
{
  pending_candidate_indices_.clear();
  std::vector< Eigen::VectorXd > requests;

  for ( std::size_t candidate_index = 0;
        candidate_index < candidates_.size(); ++candidate_index )
  {
    const RankingSelectionCandidate &candidate =
      candidates_.at( candidate_index );
    if ( !candidate.eligible )
    {
      continue;
    }

    const std::size_t current_replications =
      candidate.aggregate.evaluation.sampleCount();
    const std::size_t missing_replications =
      candidate.required_replications > current_replications
        ? candidate.required_replications - current_replications
        : 0;

    for ( std::size_t replication = 0;
          replication < missing_replications; ++replication )
    {
      requests.push_back( candidate.aggregate.parameters );
      pending_candidate_indices_.push_back( candidate_index );
    }
  }

  return requests;
}

void RinottRankingSelectionGPSMethod::collectReplications(
  const std::vector< CandidateEvaluation > &evaluations )
{
  if ( evaluations.size() != pending_candidate_indices_.size() )
  {
    throw std::invalid_argument(
      "RinottRankingSelectionGPSMethod: evaluation count does not match requested replications" );
  }

  for ( std::size_t evaluation_index = 0;
        evaluation_index < evaluations.size(); ++evaluation_index )
  {
    const CandidateEvaluation &evaluation =
      evaluations.at( evaluation_index );

    if ( !multiple_samples_warning_emitted_ &&
         evaluation.evaluation.sampleCount() > 1 )
    {
      std::cerr
        << "RinottRankingSelectionGPSMethod: each problem evaluation contains multiple samples; "
           "this method already performs repeated sampling and computes candidate means itself\n";
      multiple_samples_warning_emitted_ = true;
    }

    RankingSelectionCandidate &candidate =
      candidates_.at( pending_candidate_indices_.at( evaluation_index ) );

    if ( evaluation.status != CandidateEvaluationStatus::Succeeded )
    {
      candidate.eligible = false;
      candidate.aggregate.status = CandidateEvaluationStatus::Failed;
      continue;
    }

    // One call to the problem is one outer replication. If the problem
    // internally sampled more than once, its mean is the outer observation.
    candidate.aggregate.evaluation.appendSample( evaluation.meanValue() );
    candidate.aggregate.evaluation.appendExecutionMetrics( evaluation.evaluation.executionMetrics() );
  }

  for ( RankingSelectionCandidate &candidate : candidates_ )
  {
    candidate.aggregate.status =
      candidate.eligible &&
          candidate.aggregate.evaluation.sampleCount() ==
            candidate.required_replications
        ? CandidateEvaluationStatus::Succeeded
        : CandidateEvaluationStatus::Failed;
  }

  pending_candidate_indices_.clear();
}

bool RinottRankingSelectionGPSMethod::prepareSecondStage()
{
  if ( !candidates_.at( incumbent_index_ ).eligible )
  {
    throw ConfigurationEvaluationError(
      "RinottRankingSelectionGPSMethod: incumbent evaluation failed during ranking and selection" );
  }

  const std::size_t eligible_candidate_count =
    static_cast< std::size_t >(
      std::count_if(
        candidates_.begin(), candidates_.end(),
        []( const RankingSelectionCandidate &candidate )
        { return candidate.eligible; } ) );

  if ( eligible_candidate_count < 2 )
  {
    return false;
  }

  const double rinott_constant =
    calculateRinottConstant(
      eligible_candidate_count,
      hyperparameters_.initial_sample_size - 1,
      1.0 - significance_ );

  bool requires_second_stage = false;
  for ( RankingSelectionCandidate &candidate : candidates_ )
  {
    if ( !candidate.eligible )
    {
      continue;
    }

    const double standard_deviation =
      candidate.aggregate.evaluation
        .standardDeviationBesselCorrection();
    const double ratio =
      rinott_constant * standard_deviation / indifference_;
    const double requested_replications = std::ceil( ratio * ratio );

    if ( !std::isfinite( requested_replications ) ||
         requested_replications >
           static_cast< double >(
             std::numeric_limits< std::size_t >::max() ) )
    {
      throw ConfigurationEvaluationError(
        "RinottRankingSelectionGPSMethod: required sample size is not representable" );
    }

    candidate.required_replications =
      std::max(
        hyperparameters_.initial_sample_size,
        static_cast< std::size_t >( requested_replications ) );
    requires_second_stage =
      requires_second_stage ||
      candidate.required_replications >
        candidate.aggregate.evaluation.sampleCount();
  }

  return requires_second_stage;
}

OptimizerMethodUpdate RinottRankingSelectionGPSMethod::finishRankingSelection()
{
  RankingSelectionCandidate &incumbent =
    candidates_.at( incumbent_index_ );
  if ( incumbent.aggregate.status != CandidateEvaluationStatus::Succeeded )
  {
    throw ConfigurationEvaluationError(
      "RinottRankingSelectionGPSMethod: incumbent does not have all required replications" );
  }

  std::size_t selected_index = incumbent_index_;
  for ( std::size_t candidate_index = 0;
        candidate_index < candidates_.size(); ++candidate_index )
  {
    const RankingSelectionCandidate &candidate =
      candidates_.at( candidate_index );
    if ( candidate.aggregate.status == CandidateEvaluationStatus::Succeeded &&
         candidate.aggregate.meanValue() <
           candidates_.at( selected_index ).aggregate.meanValue() )
    {
      selected_index = candidate_index;
    }
  }

  std::vector< CandidateEvaluation > trial_evaluations;
  trial_evaluations.reserve( candidates_.size() - 1 );
  for ( std::size_t candidate_index = 0;
        candidate_index < candidates_.size(); ++candidate_index )
  {
    if ( candidate_index != incumbent_index_ )
    {
      trial_evaluations.push_back(
        candidates_.at( candidate_index ).aggregate );
    }
  }

  const CandidateEvaluation incumbent_evaluation = incumbent.aggregate;
  incumbent_evaluation_ = candidates_.at( selected_index ).aggregate;
  const std::optional< CandidateEvaluation > accepted_candidate =
    selected_index == incumbent_index_
      ? std::nullopt
      : std::optional< CandidateEvaluation >{
          candidates_.at( selected_index ).aggregate };

  // S_k belongs to the point selected as the next incumbent.
  incumbent_standard_deviation_ =
    candidates_.at( selected_index )
      .aggregate.evaluation.standardDeviationBesselCorrection();

  // The paper updates alpha and delta after every R&S invocation, therefore a
  // failed search followed by poll performs two updates in one GPS iteration.
  significance_ *= hyperparameters_.significance_decay;
  indifference_ *= hyperparameters_.indifference_decay;
  ++ranking_selection_call_;

  const Phase completed_phase = pending_gps_phase_;
  ranking_selection_phase_ = RankingSelectionPhase::Delegating;
  candidates_.clear();
  pending_candidate_indices_.clear();

  if ( completed_phase == Phase::WaitingSearch )
  {
    completeSearchStep(
      incumbent_evaluation, trial_evaluations, accepted_candidate );
    return OptimizerMethodUpdate{ accepted_candidate.has_value() };
  }
  if ( completed_phase == Phase::WaitingPoll )
  {
    completePollStep(
      incumbent_evaluation, trial_evaluations, accepted_candidate );
    return {};
  }

  throw std::logic_error(
    "RinottRankingSelectionGPSMethod: ranking and selection has no pending GPS step" );
}

std::vector< Eigen::VectorXd >
RinottRankingSelectionGPSMethod::ask()
{
  switch ( ranking_selection_phase_ )
  {
    case RankingSelectionPhase::SecondStage:
    {
      std::vector< Eigen::VectorXd > requests = buildPendingRequests();
      if ( requests.empty() )
      {
        throw std::logic_error(
          "RinottRankingSelectionGPSMethod: second stage has no pending replications" );
      }

      ranking_selection_phase_ =
        RankingSelectionPhase::WaitingSecondStage;
      return requests;
    }

    case RankingSelectionPhase::WaitingFirstStage:
    case RankingSelectionPhase::WaitingSecondStage:
      throw std::logic_error(
        "RinottRankingSelectionGPSMethod: ask called while waiting for replications" );

    case RankingSelectionPhase::Delegating:
      break;
  }

  std::vector< Eigen::VectorXd > trial_points = GPSMethod::ask();

  // Initialization remains a single ordinary GPS evaluation. Search and poll
  // batches are instead compared jointly with a freshly sampled incumbent.
  if ( phase() == Phase::WaitingInitialization )
  {
    return trial_points;
  }

  pending_gps_phase_ = phase();
  return beginRankingSelection( trial_points );
}

OptimizerMethodUpdate RinottRankingSelectionGPSMethod::tellImpl(
  const std::vector< CandidateEvaluation > &evaluations )
{
  switch ( ranking_selection_phase_ )
  {
    case RankingSelectionPhase::WaitingFirstStage:
      collectReplications( evaluations );
      if ( prepareSecondStage() )
      {
        ranking_selection_phase_ = RankingSelectionPhase::SecondStage;
        return OptimizerMethodUpdate{ false };
      }
      return finishRankingSelection();

    case RankingSelectionPhase::WaitingSecondStage:
      collectReplications( evaluations );
      return finishRankingSelection();

    case RankingSelectionPhase::Delegating:
    {
      const OptimizerMethodUpdate update = GPSMethod::tellImpl( evaluations );
      incumbent_evaluation_ = evaluations.front();
      return update;
    }

    case RankingSelectionPhase::SecondStage:
      throw std::logic_error(
        "RinottRankingSelectionGPSMethod: tell called before requesting second-stage replications" );
  }

  throw std::logic_error(
    "RinottRankingSelectionGPSMethod: invalid ranking-selection phase" );
}

void RinottRankingSelectionGPSMethod::updateBestCandidate(
  const std::vector< CandidateEvaluation > & )
{
  if ( ranking_selection_phase_ == RankingSelectionPhase::Delegating &&
       incumbent_evaluation_.status == CandidateEvaluationStatus::Succeeded )
  {
    // Rinott's incumbent is selected from complete aggregate estimates. A
    // first-stage batch stays internal while further replications are pending.
    best_candidate_evaluation_ = incumbent_evaluation_;
  }
}

OptimizerMethod::ConvergenceStatus
RinottRankingSelectionGPSMethod::convergenceStatus() const
{
  if ( !stopping_strategy_->hasCriterion() )
  {
    return ConvergenceStatus::NoInternalCriterion;
  }

  const StochasticGPSStoppingState state{
    meshSize(),
    significance_,
    indifference_,
    incumbent_standard_deviation_
  };

  return stopping_strategy_->shouldStop( state )
    ? ConvergenceStatus::Converged
    : ConvergenceStatus::InProgress;
}

void RinottRankingSelectionGPSMethod::reset()
{
  GPSMethod::reset();
  ranking_selection_phase_ = RankingSelectionPhase::Delegating;
  pending_gps_phase_ = Phase::WaitingSearch;
  candidates_.clear();
  pending_candidate_indices_.clear();
  incumbent_index_ = 0;
  incumbent_evaluation_ = CandidateEvaluation{};
  significance_ = hyperparameters_.initial_significance;
  indifference_ = hyperparameters_.initial_indifference;
  incumbent_standard_deviation_.reset();
  stopping_strategy_ =
    createStochasticGPSStoppingStrategy(
      hyperparameters_.stopping_criterion,
      hyperparameters_.stopping_mesh_size );
  ranking_selection_call_ = 0;
  multiple_samples_warning_emitted_ = false;
}
