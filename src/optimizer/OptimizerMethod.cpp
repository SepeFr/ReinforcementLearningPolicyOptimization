#include "OptimizerMethod.hpp"
#include <vector>
#include "CandidateEvaluation.hpp"
#include "InitialParametersStrategy.hpp"


OptimizerMethod::OptimizerMethod( InitialParametersStrategy *initial_parameters_strategy ) :
    initial_parameters_strategy_( initial_parameters_strategy )
{
}


OptimizerMethodUpdate OptimizerMethod::tell( const std::vector< CandidateEvaluation > &evaluations )
{
  const OptimizerMethodUpdate update = tellImpl( evaluations );
  updateBestCandidate( evaluations );
  return update;
}


const CandidateEvaluation &OptimizerMethod::bestCandidate() const { return best_candidate_evaluation_; }


OptimizerMethod::ConvergenceStatus OptimizerMethod::convergenceStatus() const
{
  return ConvergenceStatus::NoInternalCriterion;
}


void OptimizerMethod::reset()
{
  best_candidate_evaluation_ =
    CandidateEvaluation{ Eigen::VectorXd{}, ObjectiveEvaluation{}, CandidateEvaluationStatus::NotEvaluated };
}


void OptimizerMethod::setInitialParametersStrategy( InitialParametersStrategy *initial_parameters_strategy )
{
  initial_parameters_strategy_ = initial_parameters_strategy;
}


void OptimizerMethod::updateBestCandidate( const std::vector< CandidateEvaluation > &evaluations )
{
  const std::optional< CandidateEvaluation > batch_best_candidate = extractBestCandidate( evaluations );
  if ( !batch_best_candidate.has_value() )
  {
    return;
  }

  if ( best_candidate_evaluation_.status != CandidateEvaluationStatus::Succeeded )
  {
    best_candidate_evaluation_ = *batch_best_candidate;
    return;
  }

  const double batch_best_value = batch_best_candidate->meanValue();
  const double current_best_value = best_candidate_evaluation_.meanValue();

  if ( batch_best_value < current_best_value )
  {
    best_candidate_evaluation_ = *batch_best_candidate;
  }
}
