#include "optimizations_methods/nelder_mead/StochasticHeuristicNelderMeadMethod.hpp"
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

StochasticHeuristicNelderMeadMethod::StochasticHeuristicNelderMeadMethod(
  std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy ) :
    StochasticHeuristicNelderMeadMethod(
      StochasticHeuristicNelderMeadHyperparameters{}, number_of_parameters,
      initial_parameters_strategy )
{
}

StochasticHeuristicNelderMeadMethod::StochasticHeuristicNelderMeadMethod(
  const StochasticHeuristicNelderMeadHyperparameters &hyperparameters,
  std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy ) :
    NelderMeadMethod( hyperparameters, number_of_parameters, initial_parameters_strategy ),
    hyperparameters_( hyperparameters )
{
  if ( hyperparameters_.shrink_coefficient == 0.5 )
  {
    std::cerr
      << "StochasticHeuristicNelderMeadMethod: shrink coefficient 0.50 can make "
         "false convergence under noise more likely\n";
  }
}

bool StochasticHeuristicNelderMeadMethod::samePoint(
  Eigen::Ref< const Eigen::VectorXd > left, Eigen::Ref< const Eigen::VectorXd > right )
{
  return left.size() == right.size() && ( left.array() == right.array() ).all();
}

void StochasticHeuristicNelderMeadMethod::updateBestStagnation()
{
  const Eigen::VectorXd &best_parameters = simplex().front().parameters;

  if ( has_last_best_ && samePoint( best_parameters, last_best_parameters_ ) )
  {
    ++unchanged_best_cycles_;
  }
  else
  {
    last_best_parameters_ = best_parameters;
    has_last_best_ = true;
    unchanged_best_cycles_ = 1;
  }

  // Spendley's rule counts complete simplexes as iterations.
  if ( hyperparameters_.reevaluate_stagnant_best &&
       unchanged_best_cycles_ >= vertexCount() )
  {
    heuristic_phase_ = HeuristicPhase::ReevaluateStagnantBest;
  }
}

std::vector< Eigen::VectorXd >
StochasticHeuristicNelderMeadMethod::contractionConfirmationCandidates()
{
  // Snapshot the base phase before entering the heuristic waiting state.
  pending_contraction_phase_ = phase();
  contraction_worst_parameters_ = simplex().back().parameters;

  if ( pending_contraction_phase_ == Phase::OutsideContraction )
  {
    // Outside contraction is delimited by the second-worst and worst values.
    return {
      reflectedPoint(),
      simplex().at( vertexCount() - 2 ).parameters,
      simplex().back().parameters
    };
  }

  // Inside contraction compares the reflection directly with the worst value.
  return { reflectedPoint(), simplex().back().parameters };
}

std::vector< Eigen::VectorXd > StochasticHeuristicNelderMeadMethod::ask()
{
  // Heuristic requests are served first. When none is active, the ordinary
  // Nelder-Mead state machine remains responsible for candidate generation.
  switch ( heuristic_phase_ )
  {
    case HeuristicPhase::ReevaluateStagnantBest:
      heuristic_phase_ = HeuristicPhase::WaitingStagnantBest;
      return { simplex().front().parameters };

    case HeuristicPhase::ConfirmContraction:
      heuristic_phase_ = HeuristicPhase::WaitingContractionConfirmation;
      return contractionConfirmationCandidates();

    case HeuristicPhase::RefreshBestBeforeShrink:
      heuristic_phase_ = HeuristicPhase::WaitingBestBeforeShrink;
      return { simplex().front().parameters };

    case HeuristicPhase::Delegating:
      break;

    default:
      throw std::logic_error(
        "StochasticHeuristicNelderMeadMethod: ask called while waiting for evaluations" );
  }

  if ( hyperparameters_.confirm_contraction &&
       ( phase() == Phase::OutsideContraction || phase() == Phase::InsideContraction ) &&
       !contraction_confirmation_complete_ )
  {
    heuristic_phase_ = HeuristicPhase::ConfirmContraction;
    return ask();
  }

  if ( phase() == Phase::Shrink && hyperparameters_.reevaluate_best_before_shrink &&
       !shrink_refresh_complete_ )
  {
    heuristic_phase_ = HeuristicPhase::RefreshBestBeforeShrink;
    return ask();
  }

  // A completed confirmation or refresh authorizes exactly one delegated operation.
  if ( phase() == Phase::OutsideContraction || phase() == Phase::InsideContraction )
  {
    contraction_confirmation_complete_ = false;
  }
  if ( phase() == Phase::Shrink )
  {
    shrink_refresh_complete_ = false;
  }

  return NelderMeadMethod::ask();
}

void StochasticHeuristicNelderMeadMethod::handleStagnantBestEvaluation(
  const std::vector< CandidateEvaluation > &evaluations )
{
  if ( evaluations.size() != 1 )
  {
    throw std::invalid_argument(
      "StochasticHeuristicNelderMeadMethod: best-point refresh requires one evaluation" );
  }

  if ( evaluations.front().status == CandidateEvaluationStatus::Succeeded )
  {
    // Spendley's rule installs the fresh evaluation as the suspicious observation.
    simplexVertex( 0 ) = evaluations.front();
  }

  prepareIteration();
  last_best_parameters_ = simplex().front().parameters;
  has_last_best_ = true;
  unchanged_best_cycles_ = 1;
  heuristic_phase_ = HeuristicPhase::Delegating;
}

void StochasticHeuristicNelderMeadMethod::handleContractionConfirmation(
  const std::vector< CandidateEvaluation > &evaluations )
{
  // Outside contraction depends on reflection, second-worst and worst.
  // Inside contraction only compares reflection with the worst vertex.
  const std::size_t expected_evaluation_count =
    pending_contraction_phase_ == Phase::OutsideContraction ? 3 : 2;
  if ( evaluations.size() != expected_evaluation_count )
  {
    throw std::invalid_argument(
      "StochasticHeuristicNelderMeadMethod: invalid contraction-confirmation evaluation count" );
  }

  const CandidateEvaluation refreshed_reflection = evaluations.front();
  if ( pending_contraction_phase_ == Phase::OutsideContraction )
  {
    if ( evaluations.at( 1 ).status == CandidateEvaluationStatus::Succeeded )
    {
      simplexVertex( vertexCount() - 2 ) = evaluations.at( 1 );
    }
    if ( evaluations.at( 2 ).status == CandidateEvaluationStatus::Succeeded )
    {
      simplexVertex( vertexCount() - 1 ) = evaluations.at( 2 );
    }
  }
  else if ( evaluations.at( 1 ).status == CandidateEvaluationStatus::Succeeded )
  {
    simplexVertex( vertexCount() - 1 ) = evaluations.at( 1 );
  }

  prepareIteration();
  heuristic_phase_ = HeuristicPhase::Delegating;

  // Refreshed observations can reorder the simplex. The old reflection and
  // centroid remain geometrically valid only while the excluded worst vertex
  // is unchanged; otherwise a new base iteration must construct them again.
  if ( convergenceStatus() == ConvergenceStatus::Converged ||
       !samePoint( simplex().back().parameters, contraction_worst_parameters_ ) )
  {
    contraction_confirmation_complete_ = false;
    return;
  }

  processReflectionEvaluation( refreshed_reflection );
  contraction_confirmation_complete_ =
    phase() == Phase::OutsideContraction || phase() == Phase::InsideContraction;
}

void StochasticHeuristicNelderMeadMethod::handleBestBeforeShrinkEvaluation(
  const std::vector< CandidateEvaluation > &evaluations )
{
  if ( evaluations.size() != 1 )
  {
    throw std::invalid_argument(
      "StochasticHeuristicNelderMeadMethod: pre-shrink refresh requires one evaluation" );
  }

  const Eigen::VectorXd previous_best = simplex().front().parameters;
  if ( evaluations.front().status == CandidateEvaluationStatus::Succeeded )
  {
    simplexVertex( 0 ) = evaluations.front();
  }

  orderSimplex();
  heuristic_phase_ = HeuristicPhase::Delegating;

  if ( samePoint( simplex().front().parameters, previous_best ) )
  {
    // The refreshed point remains the anchor of the pending shrink.
    shrink_refresh_complete_ = true;
    setPhase( Phase::Shrink );
    return;
  }

  // A changed best invalidates the old shrink decision.
  shrink_refresh_complete_ = false;
  prepareIteration();
}

OptimizerMethodUpdate StochasticHeuristicNelderMeadMethod::tellImpl(
  const std::vector< CandidateEvaluation > &evaluations )
{
  const std::size_t previous_iteration_count = iterationCount();

  switch ( heuristic_phase_ )
  {
    case HeuristicPhase::WaitingStagnantBest:
      handleStagnantBestEvaluation( evaluations );
      return OptimizerMethodUpdate{ false };

    case HeuristicPhase::WaitingContractionConfirmation:
    {
      handleContractionConfirmation( evaluations );
      const OptimizerMethodUpdate update{
        iterationCount() > previous_iteration_count
      };
      if ( update.iteration_completed && phase() == Phase::Reflection )
      {
        updateBestStagnation();
      }
      return update;
    }

    case HeuristicPhase::WaitingBestBeforeShrink:
      handleBestBeforeShrinkEvaluation( evaluations );
      return OptimizerMethodUpdate{ false };

    case HeuristicPhase::Delegating:
      break;

    default:
      throw std::logic_error(
        "StochasticHeuristicNelderMeadMethod: tell called without pending evaluations" );
  }

  const OptimizerMethodUpdate update = NelderMeadMethod::tellImpl( evaluations );

  if ( update.iteration_completed && phase() == Phase::Reflection )
  {
    updateBestStagnation();
  }

  return update;
}

void StochasticHeuristicNelderMeadMethod::reset()
{
  NelderMeadMethod::reset();
  heuristic_phase_ = HeuristicPhase::Delegating;
  last_best_parameters_.resize( 0 );
  contraction_worst_parameters_.resize( 0 );
  pending_contraction_phase_ = Phase::OutsideContraction;
  has_last_best_ = false;
  contraction_confirmation_complete_ = false;
  shrink_refresh_complete_ = false;
  unchanged_best_cycles_ = 0;
}
