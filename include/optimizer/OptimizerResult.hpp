#ifndef OPTIMIZER_RESULT_H
#define OPTIMIZER_RESULT_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <vector>
#include "CandidateEvaluation.hpp"
#include "ExecutionMetrics.hpp"
#include "OptimizationTerminationReason.hpp"

/** @brief Final incumbent, counters, history, work, and termination state. */
struct OptimizerResult
{
  CandidateEvaluation best_candidate; ///< Best successful candidate in the problem's natural objective direction.
  std::size_t number_of_iterations; ///< Logical iterations reported complete by OptimizerMethodUpdate.
  std::size_t number_of_batches; ///< Nonempty candidate batches processed by Optimizer.
  std::size_t number_of_evaluations; ///< Attempted candidates, including rejected non-finite and out-of-bounds values.
  std::vector< CandidateEvaluation > best_parameters_history; ///< Incumbent snapshots in chronological order.
  ExecutionMetrics execution_metrics; ///< Metrics from every call made to BlackBoxProblem::evaluate().
  OptimizationTerminationReason termination_reason = OptimizationTerminationReason::None; ///< Condition that ended the run.
};

/** @} */

#endif // !OPTIMIZER_RESULT_H
