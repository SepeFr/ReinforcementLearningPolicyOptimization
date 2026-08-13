#ifndef CANDIDATE_EVALUATION_H
#define CANDIDATE_EVALUATION_H

/** @addtogroup optimizer_core_api
 * @{ */

/** @file CandidateEvaluation.hpp @brief Optimizer candidate outcome types and batch incumbent extraction. */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <optional>
#include <stdexcept>
#include <vector>
#include "ObjectiveEvaluation.hpp"

/** @brief Lifecycle state of a candidate evaluation. */
enum class CandidateEvaluationStatus
{
  NotEvaluated, ///< No objective result is available.
  Succeeded,    ///< A nonempty, finite objective evaluation is available.
  Failed        ///< Evaluation was rejected or produced an unusable result.
};

/** @brief Couples one parameter candidate with its objective outcome. */
struct CandidateEvaluation
{
  Eigen::VectorXd parameters; ///< Candidate coordinates in problem parameter order.
  ObjectiveEvaluation evaluation; ///< Raw samples and metrics associated with the candidate.
  CandidateEvaluationStatus status = CandidateEvaluationStatus::NotEvaluated; ///< Availability of a usable outcome.

  /**
   * @brief Returns the objective mean of a successful candidate.
   * @return Mean of evaluation samples.
   * @throws std::logic_error Unless status is CandidateEvaluationStatus::Succeeded.
   */
  double meanValue() const
  {
    if ( status != CandidateEvaluationStatus::Succeeded )
    {
      throw std::logic_error( "CandidateEvaluation: mean value is available only after a successful evaluation" );
    }
    return evaluation.meanValue();
  }

  /**
   * @brief Returns the number of failed samples after evaluation.
   * @return ObjectiveEvaluation::failedSampleCount().
   * @throws std::logic_error If status is CandidateEvaluationStatus::NotEvaluated.
   */
  std::size_t failedSampleCount() const
  {
    if ( status == CandidateEvaluationStatus::NotEvaluated )
    {
      throw std::logic_error( "CandidateEvaluation: failed sample count is unavailable before evaluation" );
    }
    return evaluation.failedSampleCount();
  }
};

/**
 * @brief Selects the lowest-mean successful candidate from a batch.
 * @param[in] candidates Candidate evaluations in arbitrary order.
 * @return Copy of the first lowest-mean successful candidate, or std::nullopt
 * when none succeeded.
 */
inline std::optional< CandidateEvaluation > extractBestCandidate( const std::vector< CandidateEvaluation > &candidates )
{
  std::optional< CandidateEvaluation > best_candidate;

  for ( const CandidateEvaluation &candidate : candidates )
  {
    if ( candidate.status != CandidateEvaluationStatus::Succeeded )
    {
      continue;
    }

    if ( !best_candidate.has_value() || candidate.meanValue() < best_candidate->meanValue() )
    {
      best_candidate = candidate;
    }
  }

  return best_candidate;
}

/** @} */

#endif // !CANDIDATE_EVALUATION_H
