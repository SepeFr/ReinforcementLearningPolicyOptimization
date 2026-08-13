#ifndef OPTIMIZER_NELDER_MEAD_METHOD_H
#define OPTIMIZER_NELDER_MEAD_METHOD_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <memory>
#include <utility>
#include <vector>
#include "MethodHyperparameters.hpp"
#include "OptimizerMethod.hpp"
#include "optimizations_methods/nelder_mead/NelderMeadStoppingCriterion.hpp"

class OptimizerFactory;

/**
 * @brief Implements an ordered, phase-driven Nelder--Mead simplex method.
 *
 * A dimension \f$n\f$ uses \f$n+1\f$ evaluated vertices. Each iteration
 * excludes the current worst vertex from the centroid, then moves through
 * reflection, optional expansion or contraction, and optional shrink. Trial
 * points use the common direction \f$c-x_n\f$:
 * \f[
 * x_r=c+\alpha(c-x_n),\quad
 * x_e=c+\gamma(c-x_n),\quad
 * x_{oc}=c+\rho_o(c-x_n),\quad
 * x_{ic}=c+\rho_i(c-x_n).
 * \f]
 * A shrink replaces each non-best vertex with
 * \f$x_0+\sigma(x_i-x_0)\f$. The initial-simplex evaluation establishes state
 * and is reported as an incomplete logical iteration.
 *
 * @see optimization_method_states_chapter
 */
class NelderMeadMethod : public OptimizerMethod
{
  public:
  /**
   * @brief Produces the candidate batch required by the current simplex phase.
   * @return All initial vertices, one trial point, or all non-best shrink vertices.
   * @throws std::logic_error If feedback for the previous batch is still pending.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Restores the construction simplex and clears all phase and convergence state. */
  void reset() override;
  /** @brief Reports the latest complete-simplex stopping result. @return Converged or InProgress. */
  ConvergenceStatus convergenceStatus() const override;

  protected:
  /** @brief Internal operation and pending-feedback phases. */
  enum class Phase
  {
    InitialSimplex,            ///< Ready to request every initial vertex.
    WaitingInitialSimplex,     ///< Awaiting all initial vertex evaluations.
    Reflection,                ///< Ready to construct the reflection point.
    WaitingReflection,         ///< Awaiting the reflection evaluation.
    Expansion,                 ///< Ready to construct the expansion point.
    WaitingExpansion,          ///< Awaiting the expansion evaluation.
    OutsideContraction,        ///< Ready to construct the outside contraction.
    WaitingOutsideContraction, ///< Awaiting the outside-contraction evaluation.
    InsideContraction,         ///< Ready to construct the inside contraction.
    WaitingInsideContraction,  ///< Awaiting the inside-contraction evaluation.
    Shrink,                    ///< Ready to move every non-best vertex toward the best.
    WaitingShrink              ///< Awaiting all changed shrink-vertex evaluations.
  };

  /**
   * @brief Constructs a method with default simplex settings.
   * @param[in] number_of_parameters Problem dimension \f$n\f$.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   */
  NelderMeadMethod( std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Constructs and records an unevaluated initial simplex.
   * @param[in] hyperparameters Initialization, coefficients, and convergence settings.
   * @param[in] number_of_parameters Problem dimension \f$n\f$; creates \f$n+1\f$ vertices.
   * @param[in] initial_parameters_strategy Non-owning strategy used by applicable initializers.
   * @throws std::invalid_argument If the scale, dimensions, bounds, attempts, or provided simplex are invalid.
   * @throws ConfigurationEvaluationError If no full-rank Latin hypercube simplex is found.
   */
  NelderMeadMethod( const NelderMeadHyperparameters &hyperparameters, std::size_t number_of_parameters,
                    InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Dispatches ordered feedback according to the pending phase.
   * @param[in] evaluations Exact batch requested by the preceding ask().
   * @return Whether this feedback completed a logical simplex iteration.
   * @throws std::invalid_argument If feedback cardinality differs from the phase request.
   * @throws ConfigurationEvaluationError If an initial or changed shrink vertex failed.
   * @throws std::logic_error If no feedback is pending.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;

  /** @brief Returns the current state-machine phase. @return Current phase. */
  Phase phase() const;
  /** @brief Replaces the current phase for a derived state machine. @param[in] phase New operation or waiting phase. */
  void setPhase( Phase phase );
  /** @brief Returns simplex cardinality. @return Problem dimension plus one. */
  std::size_t vertexCount() const;
  /** @brief Returns completed simplex iterations. @return Count excluding initial evaluation. */
  std::size_t iterationCount() const;

  /** @brief Returns the current evaluated simplex. @return Read-only vertices in the current stored order. */
  const std::vector< CandidateEvaluation > &simplex() const;
  /**
   * @brief Provides mutable access to one simplex vertex for stochastic extensions.
   * @param[in] index Stored vertex index.
   * @return Mutable vertex reference.
   * @throws std::out_of_range If @p index is invalid.
   */
  CandidateEvaluation &simplexVertex( std::size_t index );
  /** @brief Returns the last requested reflection coordinates. @return Read-only owned vector. */
  const Eigen::VectorXd &reflectedPoint() const;
  /** @brief Returns the last reflection feedback. @return Read-only candidate evaluation. */
  const CandidateEvaluation &reflectedCandidate() const;

  /**
   * @brief Sorts the successful simplex by increasing mean and refreshes cached values.
   * @pre Every simplex vertex is successful.
   * @post simplex().front() is best and simplex().back() is worst.
   */
  void orderSimplex();
  /**
   * @brief Orders a complete simplex, tests convergence, and prepares reflection.
   * @post A nonconverged simplex has a centroid and Phase::Reflection.
   */
  void prepareIteration();
  /** @brief Extension point invoked after a simplex iteration completes. */
  virtual void prepareNextIteration();
  /**
   * @brief Selects the post-reflection phase from one evaluation.
   * @param[in] evaluation Reflection feedback; failure selects inside contraction.
   */
  void processReflectionEvaluation( const CandidateEvaluation &evaluation );
  /**
   * @brief Replaces the worst vertex and starts the next iteration.
   * @param[in] replacement Successful candidate selected by the current transition.
   */
  void completeIteration( const CandidateEvaluation &replacement );
  /** @brief Advances the iteration counter without replacing a vertex. */
  void completeIterationWithoutReplacement();

  private:
  /** Allows the factory to invoke protected assembly constructors. */
  friend class OptimizerFactory;

  /** @brief Dispatches the configured initial-simplex construction and saves it for reset(). */
  void initializeSimplex();
  /** @brief Builds \f$x_0\f$ followed by \f$x_0+s e_i\f$ for every coordinate. */
  void initializeClassicalLocalSimplex();
  /**
   * @brief Tries local Latin hypercube directions until affine rank is full.
   * @param[in] configuration Attempt limit and seed.
   */
  void initializeLatinHypercubeLocalSimplex( const LatinHypercubeLocalSimplexConfiguration &configuration );
  /**
   * @brief Tries Latin hypercube vertices mapped into a configured box.
   * @param[in] configuration Bounds, attempt limit, and seed.
   */
  void initializeLatinHypercubeGlobalSimplex( const LatinHypercubeGlobalSimplexConfiguration &configuration );
  /** @brief Consumes vertexCount() vectors from the initialization strategy. */
  void initializeProvidedSimplex();
  /** @brief Tests rank of vertex differences from the first vertex. @return Whether affine rank equals problem dimension. */
  bool hasFullAffineRank() const;
  /** @brief Copies current vertex coordinates. @return Parameters in stored simplex order. */
  std::vector< Eigen::VectorXd > simplexParameters() const;

  /** Current operation or pending-feedback phase. */
  Phase phase_ = Phase::InitialSimplex;
  /** Construction-time unevaluated simplex restored by reset(). */
  std::vector< CandidateEvaluation > initial_simplex_;
  /** Current vertex parameters and evaluations; stochastic subclasses extend these aggregates. */
  std::vector< CandidateEvaluation > simplex_;
  /** Last reflection feedback retained across expansion or contraction. */
  CandidateEvaluation reflected_candidate_;
  /** Last requested reflection coordinates. */
  Eigen::VectorXd reflected_point_;
  /** Last requested expansion coordinates. */
  Eigen::VectorXd expansion_point_;
  /** Last requested outside-contraction coordinates. */
  Eigen::VectorXd outside_contraction_point_;
  /** Last requested inside-contraction coordinates. */
  Eigen::VectorXd inside_contraction_point_;

  /** Cached ordered simplex means, matching simplex_ indices. */
  std::vector< double > f_;
  /** Mean of every vertex except the worst. */
  Eigen::VectorXd centroid_;

  /** Fixed simplex cardinality \f$n+1\f$. */
  std::size_t number_of_vertices_;
  /** Copied simplex configuration and coefficients. */
  NelderMeadHyperparameters hyperparameters_;
  /** Owned convergence strategy, recreated by reset(). */
  std::unique_ptr< NelderMeadStoppingStrategy > stopping_strategy_;
  /** Cached current best mean. */
  double f_best_ = 0.0;
  /** Cached successful reflection mean. */
  double f_r_ = 0.0;
  /** Completed post-initialization simplex iterations. */
  std::size_t iteration_counter_ = 0;
  /** Result of the most recent complete-simplex stopping test. */
  bool has_converged_ = false;
};

/** @} */

#endif // !OPTIMIZER_NELDER_MEAD_METHOD_H
