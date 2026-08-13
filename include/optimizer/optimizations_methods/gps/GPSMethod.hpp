#ifndef OPTIMIZER_GPS_METHOD_H
#define OPTIMIZER_GPS_METHOD_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "MethodHyperparameters.hpp"
#include "ObjectiveEvaluation.hpp"
#include "OptimizerMethod.hpp"
#include "optimizations_methods/gps/GPSPollStrategy.hpp"
#include "optimizations_methods/gps/GPSSearchStrategy.hpp"

class OptimizerFactory;

/**
 * @brief Implements generalized pattern search on a scaled integer mesh.
 *
 * For incumbent \f$x_k\f$, mesh size \f$\delta_k\f$, and
 * \f$D=GZ\f$, candidates have the form \f$x_k+\delta_k D y\f$. A search
 * phase may produce any finite batch. A strict search improvement completes
 * the iteration and skips poll. Otherwise the poll evaluates selected columns
 * of \f$D\f$. Success expands the mesh by division by
 * mesh_size_adjustment; poll failure contracts it by multiplication.
 *
 * @see optimization_method_states_chapter
 */
class GPSMethod : public OptimizerMethod
{
  public:
  /**
   * @brief Produces initialization, optional search, or poll candidates.
   * @return Candidate vectors in phase-defined order.
   * @throws std::logic_error If feedback for a prior batch is pending.
   * @throws std::invalid_argument If initial parameters or generating-matrix settings are invalid.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Clears incumbent and mesh state and resets owned search and poll strategies. */
  void reset() override;

  protected:
  /** @brief GPS operation and pending-feedback phases. */
  enum class Phase
  {
    Initialization,        ///< Ready to request the initial candidate and build directions.
    WaitingInitialization, ///< Awaiting initial-candidate feedback.
    Search,                ///< Ready to invoke the optional search strategy.
    WaitingSearch,         ///< Awaiting a nonempty search batch.
    Poll,                  ///< Ready to construct poll points from selected directions.
    WaitingPoll            ///< Awaiting poll feedback.
  };

  /**
   * @brief Constructs GPS with default hyperparameters.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   */
  explicit GPSMethod( InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Creates and owns configured search and poll strategies.
   * @param[in] hyperparameters Mesh, direction, search, poll, and convergence settings.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   * @throws std::invalid_argument If mesh values or a search-model setting is invalid.
   */
  GPSMethod( const GeneralizedPatternSearchHyperparameters &hyperparameters,
             InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Applies feedback for the pending GPS phase.
   * @param[in] evaluations Nonempty feedback batch.
   * @return Initialization and unsuccessful search are incomplete iterations;
   * successful search and every poll complete one iteration.
   * @throws std::invalid_argument If feedback is empty.
   * @throws ConfigurationEvaluationError If initialization failed.
   * @throws std::logic_error If no feedback is pending.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;
  /**
   * @brief Tests the strict mesh threshold after at least one completed iteration.
   * @return NoInternalCriterion when the threshold is zero; otherwise Converged
   * exactly when `k_ > 0 && delta_k_ < stopping_mesh_size`.
   */
  OptimizerMethod::ConvergenceStatus convergenceStatus() const override;

  /** @brief Returns the current state-machine phase. @return Current phase. */
  Phase phase() const;
  /** @brief Builds a successful one-sample view of the incumbent. @return Candidate containing x_k_ and f_x_k_. */
  CandidateEvaluation currentCandidate() const;
  /** @brief Returns the current mesh size. @return \f$\delta_k\f$. */
  double meshSize() const;

  /**
   * @brief Reports search observations and applies a strict search improvement.
   * @param[in] incumbent Incumbent before search.
   * @param[in] trial_evaluations Complete ordered search feedback.
   * @param[in] accepted_candidate Best strict improvement, if present.
   * @post Success expands the mesh, increments k_, and selects Search; failure selects Poll without incrementing k_.
   */
  void completeSearchStep(
    const CandidateEvaluation &incumbent,
    const std::vector< CandidateEvaluation > &trial_evaluations,
    const std::optional< CandidateEvaluation > &accepted_candidate );

  /**
   * @brief Reports poll observations and completes the GPS iteration.
   * @param[in] incumbent Incumbent before poll.
   * @param[in] trial_evaluations Complete ordered poll feedback.
   * @param[in] accepted_candidate Best strict improvement, if present.
   * @post Mesh expands on success and contracts on failure; k_ increments and phase becomes Search.
   */
  void completePollStep(
    const CandidateEvaluation &incumbent,
    const std::vector< CandidateEvaluation > &trial_evaluations,
    const std::optional< CandidateEvaluation > &accepted_candidate );

  private:
  /** Allows the factory to invoke protected assembly constructors. */
  friend class OptimizerFactory;

  /**
   * @brief Constructs \f$D=GZ\f$ after initial dimension is known.
   * @throws std::invalid_argument If dimension is zero, G is mismatched,
   * non-finite, or singular, or the positive-basis variant is invalid.
   */
  void constructDMatrix();

  /** Copied mesh, direction, strategy, and threshold settings. */
  GeneralizedPatternSearchHyperparameters hyperparameters_;
  /** Owned optional search strategy. */
  std::unique_ptr< GPSSearchStrategy > search_strategy_;
  /** Owned poll-direction strategy. */
  std::unique_ptr< GPSPollStrategy > poll_strategy_;

  /** Current incumbent coordinates \f$x_k\f$. */
  Eigen::VectorXd x_k_;
  /** Current incumbent minimization mean \f$f(x_k)\f$. */
  double f_x_k_ = 0.0;
  /** Complete real direction matrix \f$D=GZ\f$. */
  Eigen::MatrixXd D_;
  /** Current positive mesh size \f$\delta_k\f$. */
  double delta_k_ = 0.0;
  /** Completed GPS iteration count. */
  std::size_t k_ = 0;
  /** Current operation or pending-feedback phase. */
  Phase phase_ = Phase::Initialization;
};

/** @} */

#endif // !OPTIMIZER_GPS_METHOD_H
