#ifndef OPTIMIZER_CHANG_STOCHASTIC_NELDER_MEAD_METHOD_H
#define OPTIMIZER_CHANG_STOCHASTIC_NELDER_MEAD_METHOD_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <random>
#include <vector>
#include "MethodHyperparameters.hpp"
#include "optimizations_methods/nelder_mead/NelderMeadMethod.hpp"

class OptimizerFactory;

/**
 * @brief Adds cumulative replication and adaptive random search to Nelder--Mead.
 *
 * At iteration \f$k\f$, every logical candidate targets
 * \f$N_k=\max(N_{min},\lceil\sqrt{k+1}\rceil)\f$ outer evaluations. The mean
 * of each BlackBoxProblem evaluation becomes one outer replication. Existing
 * simplex aggregates retain prior replications and request only the missing
 * count. A contraction failure enters adaptive random search and repeats until
 * a candidate mean is no greater than the current worst.
 *
 * @see optimization_method_states_chapter
 */
class ChangStochasticNelderMeadMethod final : public NelderMeadMethod
{
  public:
  /**
   * @brief Expands logical candidates into repeated physical evaluations.
   * @return Flat batch grouped by logical candidate, with each candidate repeated N_k times.
   * @throws std::logic_error If refresh or adaptive-search feedback is pending.
   * @throws ConfigurationEvaluationError If local adaptive search cannot form valid fitness or geometry.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Resets base state, cumulative requests, random state, and warning state. */
  void reset() override;

  protected:
  /**
   * @brief Aggregates physical feedback and advances the active overlay or base phase.
   * @param[in] evaluations Flat feedback matching the preceding repeated request order.
   * @return Whether one logical simplex or accepted adaptive-search iteration completed.
   * @throws std::invalid_argument If physical feedback count differs from the request.
   * @throws std::logic_error If no feedback is pending or a refreshed simplex has wrong size.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;
  /**
   * @brief Sets the method incumbent from cumulative current-simplex estimates.
   * @param[in] evaluations Physical feedback; cumulative simplex state is used directly.
   */
  void updateBestCandidate( const std::vector< CandidateEvaluation > &evaluations ) override;
  /** @brief Schedules missing simplex replications before preparing the next geometric iteration. */
  void prepareNextIteration() override;

  private:
  /** @brief Maps one logical candidate to cumulative state and its next physical request count. */
  struct PendingCandidate
  {
    CandidateEvaluation aggregate; ///< Existing cumulative outer-replication observations.
    std::size_t requested_evaluations = 0; ///< Physical feedback entries consumed by the next tell().
  };

  /** @brief Overlay phases for replication refresh and adaptive random search. */
  enum class ChangPhase
  {
    Delegating,                  ///< Base Nelder--Mead supplies logical candidates.
    SimplexRefresh,              ///< Ready to request missing replications for all vertices.
    WaitingSimplexRefresh,       ///< Awaiting the flattened refresh batch.
    AdaptiveRandomSearch,        ///< Ready to generate one local or global logical candidate.
    WaitingAdaptiveRandomSearch  ///< Awaiting repeated adaptive-candidate evaluations.
  };

  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /**
   * @brief Constructs the method with default Chang settings.
   * @param[in] number_of_parameters Problem dimension.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   */
  ChangStochasticNelderMeadMethod(
    std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Constructs the base simplex, copies Chang settings, and seeds random search.
   * @param[in] hyperparameters Base coefficients, replication schedule, bounds, probability, and seed.
   * @param[in] number_of_parameters Problem dimension.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   * @throws std::invalid_argument If minimum sample size is zero or global-search probability is outside `(0,1)`.
   */
  ChangStochasticNelderMeadMethod(
    const ChangStochasticNelderMeadHyperparameters &hyperparameters,
    std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy );

  /** @brief Computes \f$N_k\f$ for the current base iteration. @return Target cumulative replication count. */
  std::size_t targetSampleCount() const;
  /**
   * @brief Requests missing replications for each stored simplex vertex.
   * @return Flat vertex-major request batch.
   */
  std::vector< Eigen::VectorXd > requestSimplexRefresh();
  /**
   * @brief Repeats every new logical candidate targetSampleCount() times.
   * @param[in] logical_candidates New candidates in base-method order.
   * @return Flat candidate-major physical request batch.
   */
  std::vector< Eigen::VectorXd >
  requestNewCandidates( const std::vector< Eigen::VectorXd > &logical_candidates );
  /**
   * @brief Converts physical results into cumulative logical evaluations.
   * @param[in] evaluations Feedback in pending candidate-major order.
   * @return One aggregate per pending logical candidate.
   * @note If one physical evaluation already contains multiple objective
   * samples, a warning is emitted once and its mean still counts as one outer replication.
   */
  std::vector< CandidateEvaluation >
  aggregatePendingEvaluations( const std::vector< CandidateEvaluation > &evaluations );
  /** @brief Writes refreshed cumulative aggregates back into the simplex. @param[in] evaluations Flattened refresh feedback. */
  void handleSimplexRefresh( const std::vector< CandidateEvaluation > &evaluations );

  /** @brief Selects a global or local adaptive candidate by configured probability. @return Candidate within configured bounds. */
  Eigen::VectorXd generateAdaptiveRandomCandidate();
  /** @brief Samples each coordinate uniformly within configured bounds. @return Global candidate. */
  Eigen::VectorXd generateGlobalAdaptiveRandomCandidate();
  /**
   * @brief Samples uniformly in a bounded ball around an inverse-objective-selected vertex.
   * @return Finite local candidate within configured bounds.
   * @throws ConfigurationEvaluationError If simplex means are non-positive or the nearest-vertex radius is degenerate.
   */
  Eigen::VectorXd generateLocalAdaptiveRandomCandidate();
  /**
   * @brief Accepts an adaptive candidate no worse than the current worst, or schedules another.
   * @param[in] evaluations Repeated physical feedback for one candidate.
   * @return Completed iteration on acceptance; incomplete update on retry.
   */
  OptimizerMethodUpdate handleAdaptiveRandomSearch( const std::vector< CandidateEvaluation > &evaluations );

  /** Copied base, replication, search, bounds, and seed configuration. */
  ChangStochasticNelderMeadHyperparameters hyperparameters_;
  /** Current overlay phase. */
  ChangPhase chang_phase_ = ChangPhase::Delegating;
  /** Logical candidates and physical counts pending aggregation. */
  std::vector< PendingCandidate > pending_candidates_;

  /** Random engine used by global/local adaptive search. */
  std::mt19937_64 generator_;
  /** Uniform distribution on \f$[0,1)\f$ used for branch, radius, and direction logic. */
  std::uniform_real_distribution< double > unit_distribution_{ 0.0, 1.0 };
  /** Prevents repeated multiple-inner-sample diagnostics. */
  bool multiple_samples_warning_emitted_ = false;
};

/** @} */

#endif // !OPTIMIZER_CHANG_STOCHASTIC_NELDER_MEAD_METHOD_H
