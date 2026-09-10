#ifndef OPTIMIZER_H
#define OPTIMIZER_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <memory>
#include <vector>
#include "BlackBoxProblem.hpp"
#include "CandidateEvaluation.hpp"
#include "InitialParametersStrategy.hpp"
#include "OptimizationStoppingCriterion.hpp"
#include "OptimizerConfiguration.hpp"
#include "OptimizerMethod.hpp"
#include "OptimizerResult.hpp"

/**
 * @brief Orchestrates a BlackBoxProblem and an OptimizerMethod to termination.
 *
 * The optimizer owns the problem, method, and initialization strategy. It
 * validates each candidate, invokes the objective, converts maximization
 * samples to minimization values for the method, and restores their natural
 * sign in OptimizerResult.
 *
 * @see optimization_lifecycle_chapter
 */
class Optimizer
{
  public:
  virtual ~Optimizer() = default;

  /**
   * @brief Takes ownership of all components required by optimize().
   * @param[in] problem Objective and feasibility contract; must be non-null.
   * @param[in] method Ask--tell algorithm; must be non-null.
   * @param[in] configuration Stopping and reproducibility settings stored by value.
   * @param[in] initial_parameters_strategy Optional owned strategy. A
   * RandomInitialization using the problem's optional bounds is created when null.
   * @throws std::invalid_argument If @p problem or @p method is null.
   */
  Optimizer( std::unique_ptr< BlackBoxProblem > problem, std::unique_ptr< OptimizerMethod > method,
             OptimizerConfiguration configuration,
             std::unique_ptr< InitialParametersStrategy > initial_parameters_strategy = nullptr );

  /**
   * @brief Runs complete ask--evaluate--tell cycles until a criterion fires.
   *
   * A candidate with the wrong dimension causes an exception. A non-finite or
   * out-of-bounds candidate becomes a failed evaluation and still consumes one
   * evaluation-budget unit. A budget-limited partial batch contributes its
   * evaluations and metrics to the result. `tell()` receives complete batches.
   *
   * @return Final successful incumbent, history, counters, metrics, and reason.
   * @throws std::invalid_argument If a method requiring an external stopping
   * criterion receives zero configured strategies, or if a candidate dimension is wrong.
   * @throws std::runtime_error If ask() returns an empty batch.
   * @throws ConfigurationEvaluationError If the run contains zero successful candidates.
   * @note Exceptions produced by the objective and concrete method propagate.
   */
  OptimizerResult optimize();

  protected:
  /** Evaluates one ask() batch. Derived optimizers may parallelize this operation. */
  virtual std::vector< CandidateEvaluation >
  evaluateCandidates( const std::vector< Eigen::VectorXd > &candidates );

  /** Applies the standard validation, evaluation, and direction-normalization rules. */
  static CandidateEvaluation evaluateCandidate( BlackBoxProblem &problem, const Eigen::VectorXd &candidate );

  /** Provides derived optimizers with the primary objective instance. */
  BlackBoxProblem &problem() { return *problem_; }

  /** Provides derived optimizers with the active ask--tell method. */
  OptimizerMethod &method() { return *method_; }


  private:
  /** Owned objective; destroyed after the method and strategy fields below. */
  std::unique_ptr< BlackBoxProblem > problem_;
  /** Owned strategy referenced non-owningly by method_. */
  std::unique_ptr< InitialParametersStrategy > initial_parameters_strategy_;
  /** Owned ask--tell method. */
  std::unique_ptr< OptimizerMethod > method_;
  /** Copied run configuration. */
  OptimizerConfiguration configuration_;
};


/** @} */

#endif // !OPTIMIZER_H
