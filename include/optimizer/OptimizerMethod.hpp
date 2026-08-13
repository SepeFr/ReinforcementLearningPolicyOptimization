#ifndef OPTIMIZER_METHOD_H
#define OPTIMIZER_METHOD_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <eigen3/Eigen/Core>
#include <vector>
#include "CandidateEvaluation.hpp"
#include "InitialParametersStrategy.hpp"

class Optimizer;

/** @brief State-transition summary returned after method feedback. */
struct OptimizerMethodUpdate
{
  bool iteration_completed = true; ///< Whether feedback completed one logical algorithm iteration.
};

/**
 * @brief Ask--tell interface implemented by every optimization algorithm.
 *
 * ask() produces an ordered batch. The matching tell() receives evaluations
 * in that exact order and updates both algorithm-specific state and the global
 * lowest-mean incumbent. All methods operate in minimization space; Optimizer
 * performs direction normalization for maximizing problems.
 *
 * @see optimization_lifecycle_chapter
 */
class OptimizerMethod
{
  public:
  /** @brief State of a method-owned convergence test. */
  enum class ConvergenceStatus
  {
    NoInternalCriterion, ///< The method relies on external stopping criteria.
    InProgress,          ///< An internal criterion exists and is not satisfied.
    Converged            ///< The internal criterion is satisfied.
  };

  /**
   * @brief Binds a non-owning initialization strategy.
   * @param[in] initial_parameters_strategy Strategy used by concrete ask()
   * implementations. Optimizer may replace this pointer before optimization.
   * @note The strategy must outlive this method while ask() can use it.
   */
  explicit OptimizerMethod( InitialParametersStrategy *initial_parameters_strategy );
  /** @brief Enables destruction through the method interface. */
  virtual ~OptimizerMethod() = default;

  /**
   * @brief Returns candidates requested by the current algorithmic phase.
   *
   * The returned order is retained until matching feedback and determines how
   * evaluations are interpreted by tell(). Concrete methods reject a second
   * ask while a batch is pending when their state machine requires it.
   *
   * @return Candidate parameter vectors in pending-evaluation order.
   */
  virtual std::vector< Eigen::VectorXd > ask() = 0;
  /**
   * @brief Applies ordered feedback and updates the global method incumbent.
   * @param[in] evaluations One evaluation per pending candidate, in ask() order.
   * @return Whether the concrete state transition completed a logical iteration.
   * @post tellImpl() runs before updateBestCandidate().
   */
  OptimizerMethodUpdate tell( const std::vector< CandidateEvaluation > &evaluations );

  /**
   * @brief Returns the lowest-mean successful candidate observed by this method.
   * @return Read-only incumbent; its status is NotEvaluated until a candidate succeeds.
   */
  const CandidateEvaluation &bestCandidate() const;

  /** @brief Reports method-owned convergence. @return NoInternalCriterion by default. */
  virtual ConvergenceStatus convergenceStatus() const;
  /**
   * @brief Clears the global incumbent before a new optimization run.
   * @post bestCandidate().status is CandidateEvaluationStatus::NotEvaluated.
   */
  virtual void reset();

  protected:
  /**
   * @brief Applies feedback to the concrete algorithm state machine.
   * @param[in] evaluations Ordered results for the most recent ask() batch.
   * @return Logical-iteration completion state.
   */
  virtual OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) = 0;
  /**
   * @brief Updates the minimization incumbent from successful feedback.
   * @param[in] evaluations Candidate batch to inspect.
   * @post The first strictly lower batch mean replaces the current incumbent.
   */
  virtual void updateBestCandidate( const std::vector< CandidateEvaluation > &evaluations );

  /** Non-owning strategy pointer established by Optimizer. */
  InitialParametersStrategy *initial_parameters_strategy_;
  /** Lowest-mean successful candidate seen through tell(). */
  CandidateEvaluation best_candidate_evaluation_;

  private:
  /** Grants Optimizer permission to replace the non-owning strategy pointer. */
  friend class Optimizer;

  /**
   * @brief Rebinds the non-owning initialization strategy.
   * @param[in] initial_parameters_strategy Strategy owned by Optimizer.
   */
  void setInitialParametersStrategy( InitialParametersStrategy *initial_parameters_strategy );
};

/** @} */

#endif // !OPTIMIZER_METHOD_H
