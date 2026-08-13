#ifndef BLACK_BOX_PROBLEM_H
#define BLACK_BOX_PROBLEM_H

/**
 * @defgroup optimizer_core_api Optimization Core
 * @brief Objective contracts, optimizer orchestration, initialization, neighborhoods, budgets, and stopping.
 * @{
 */

/** @file BlackBoxProblem.hpp @brief Black-box objective direction, dimensions, and bounds. */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <functional>
#include <optional>
#include "ObjectiveEvaluation.hpp"

/** @brief Natural direction in which a problem's objective improves. */
enum class OptimizationDirection
{
  Minimize, ///< Lower objective means are better.
  Maximize  ///< Higher objective means are better; Optimizer negates samples internally.
};

/**
 * @brief Objective interface with fixed dimensionality and optional box bounds.
 *
 * Bounds are inclusive. Optimizer reports non-finite or out-of-bounds
 * candidates as failed evaluations without calling evaluate().
 */
class BlackBoxProblem
{
  public:
  /** @brief Enables destruction through the problem interface. */
  virtual ~BlackBoxProblem() = default;

  /**
   * @brief Constructs an unbounded problem.
   * @param[in] parameter_count Required candidate cardinality.
   * @param[in] direction Natural objective direction.
   * @throws std::overflow_error If @p parameter_count cannot be represented as Eigen::Index.
   */
  explicit BlackBoxProblem( std::size_t parameter_count,
                            OptimizationDirection direction = OptimizationDirection::Minimize );

  /**
   * @brief Constructs a problem with lower and upper bounds.
   * @param[in] lower_bound Inclusive lower coordinate limits.
   * @param[in] upper_bound Inclusive upper coordinate limits in matching order.
   * @param[in] direction Natural objective direction.
   * @throws std::overflow_error If the vector dimension cannot be represented as size_t.
   * @throws std::invalid_argument If bound lengths differ or a lower coordinate exceeds its upper coordinate.
   */
  BlackBoxProblem( Eigen::Ref< const Eigen::VectorXd > lower_bound, Eigen::Ref< const Eigen::VectorXd > upper_bound,
                   OptimizationDirection direction = OptimizationDirection::Minimize );

  /**
   * @brief Constructs a problem with one side of its box bounds.
   * @param[in] bound Inclusive coordinate limits.
   * @param[in] is_lower_bound Stores @p bound as lower limits when `true`, upper limits otherwise.
   * @param[in] direction Natural objective direction.
   * @throws std::overflow_error If the vector dimension cannot be represented as size_t.
   */
  BlackBoxProblem( Eigen::Ref< const Eigen::VectorXd > bound, bool is_lower_bound,
                   OptimizationDirection direction = OptimizationDirection::Minimize );

  /**
   * @brief Evaluates one feasible candidate.
   * @param[in] parameters Candidate coordinates in problem order.
   * @return One or more raw objective samples and associated metrics.
   * @pre @p parameters has parametersCount() finite components and satisfies withinBounds().
   */
  virtual ObjectiveEvaluation evaluate( Eigen::Ref< const Eigen::VectorXd > parameters ) = 0;

  /** @brief Returns required candidate cardinality. @return Number of problem parameters. */
  std::size_t parametersCount() const;
  /** @brief Returns the natural objective direction. @return Direction supplied at construction. */
  OptimizationDirection direction() const;
  /** @brief Returns inclusive lower limits. @return Owned optional vector; empty for no lower bound. */
  const std::optional< Eigen::VectorXd > &lowerBound() const;
  /** @brief Returns inclusive upper limits. @return Owned optional vector; empty for no upper bound. */
  const std::optional< Eigen::VectorXd > &upperBound() const;

  /**
   * @brief Tests candidate dimensionality and inclusive box feasibility.
   * @param[in] parameters Candidate to test.
   * @return `true` when the dimension matches and every configured bound is satisfied.
   */
  bool withinBounds( Eigen::Ref< const Eigen::VectorXd > parameters ) const;

  private:
  /** Required number of candidate coordinates. */
  std::size_t parameters_count_;
  /** Inclusive lower limits, when configured. */
  std::optional< Eigen::VectorXd > lower_bound_;
  /** Inclusive upper limits, when configured. */
  std::optional< Eigen::VectorXd > upper_bound_;
  /** Natural direction of objective improvement. */
  OptimizationDirection direction_;
};

/** @} */

#endif // !BLACK_BOX_PROBLEM_H
