#ifndef OPTIMIZER_RINOTT_RANKING_SELECTION_GPS_METHOD_H
#define OPTIMIZER_RINOTT_RANKING_SELECTION_GPS_METHOD_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <memory>
#include <optional>
#include <vector>
#include "MethodHyperparameters.hpp"
#include "optimizations_methods/gps/GPSMethod.hpp"

class OptimizerFactory;

/**
 * @brief Wraps every GPS search and poll in two-stage ranking and selection.
 *
 * Exact duplicate trial points and the incumbent are compared once. Stage one
 * requests `initial_sample_size` outer replications per alternative. For each
 * eligible alternative \f$i\f$, stage two targets
 * \f[
 * N_i=\max\left(n_0,\left\lceil
 *       (h S_i/\delta_r)^2\right\rceil\right),
 * \f]
 * where \f$h\f$ is the numerically computed Rinott constant for confidence
 * \f$1-\alpha_r\f$, \f$S_i\f$ is Bessel-corrected stage-one deviation, and
 * \f$\delta_r\f$ is the current indifference width. Each problem evaluation's
 * mean counts as one outer replication. Failed replication makes its
 * alternative ineligible; incumbent failure aborts the selection.
 *
 * @see optimization_method_states_chapter
 */
class RinottRankingSelectionGPSMethod final : public GPSMethod
{
  public:
  /**
   * @brief Delegates GPS initialization or requests the active selection stage.
   * @return One initialization point or a flattened candidate-major replication batch.
   * @throws std::logic_error If stage feedback is pending or a scheduled second stage has no requests.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Resets GPS, sampling phases, statistical schedule, aggregates, and stopping strategy. */
  void reset() override;

  protected:
  /**
   * @brief Collects one selection stage or delegates initialization feedback.
   * @param[in] evaluations Ordered physical replication feedback.
   * @return GPS logical-iteration completion state.
   * @throws std::invalid_argument If feedback count differs from requested replications.
   * @throws ConfigurationEvaluationError If the incumbent fails or a numerical sampling calculation fails.
   * @throws std::logic_error For an invalid ask--tell phase.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;
  /**
   * @brief Exposes only a complete selected aggregate as the method incumbent.
   * @param[in] evaluations Physical feedback; the selected aggregate is used.
   */
  void updateBestCandidate( const std::vector< CandidateEvaluation > &evaluations ) override;
  /**
   * @brief Applies the configured stochastic stopping strategy.
   * @return NoInternalCriterion for the disabled strategy, otherwise Converged or InProgress.
   */
  OptimizerMethod::ConvergenceStatus convergenceStatus() const override;

  private:
  /** @brief Cumulative state for one ranking-and-selection alternative. */
  struct RankingSelectionCandidate
  {
    CandidateEvaluation aggregate; ///< Outer-replication means and accumulated metrics.
    std::size_t required_replications = 0; ///< Current stage's total target sample count.
    bool eligible = true; ///< Cleared permanently when any requested replication fails.
  };

  /** @brief Overlay phases around one GPS search or poll batch. */
  enum class RankingSelectionPhase
  {
    Delegating,         ///< GPS handles initialization or supplies a new logical batch.
    WaitingFirstStage,  ///< Awaiting n_0 replications for each alternative.
    SecondStage,        ///< Required totals computed and ready to request missing samples.
    WaitingSecondStage  ///< Awaiting all missing second-stage replications.
  };

  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /**
   * @brief Constructs default stochastic GPS.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   */
  explicit RinottRankingSelectionGPSMethod(
    InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Constructs GPS, stopping strategy, and initial statistical schedule.
   * @param[in] hyperparameters GPS, sampling, decay, and stopping settings.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   * @throws std::invalid_argument If a sample size, probability, positive scale,
   * decay, or stopping-configuration constraint fails.
   */
  RinottRankingSelectionGPSMethod(
    const RinottRankingSelectionGPSHyperparameters &hyperparameters,
    InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Deduplicates trial points, appends the incumbent, and starts stage one.
   * @param[in] trial_points GPS search or poll points in generation order.
   * @return Flattened first-stage requests grouped by retained alternative.
   */
  std::vector< Eigen::VectorXd >
  beginRankingSelection( const std::vector< Eigen::VectorXd > &trial_points );
  /**
   * @brief Builds missing-replication requests for all eligible alternatives.
   * @return Flat candidate-major batch and matching internal candidate-index map.
   */
  std::vector< Eigen::VectorXd > buildPendingRequests();
  /**
   * @brief Appends successful physical means and marks failed alternatives ineligible.
   * @param[in] evaluations Results matching the pending candidate-index map.
   * @note A one-time stderr warning is emitted if an outer evaluation already contains multiple samples.
   */
  void collectReplications( const std::vector< CandidateEvaluation > &evaluations );
  /**
   * @brief Computes the Rinott constant and each eligible alternative's total target.
   * @return Whether any eligible alternative needs more replications.
   * @throws ConfigurationEvaluationError If the incumbent failed or numerical
   * quadrature, bracketing, or sample-count conversion fails.
   */
  bool prepareSecondStage();
  /**
   * @brief Selects the lowest aggregate mean and completes the pending GPS step.
   * @return Search or poll completion state.
   * @post Significance and indifference are multiplied by their decay factors.
   */
  OptimizerMethodUpdate finishRankingSelection();

  /**
   * @brief Tests exact coordinate identity for deduplication.
   * @param[in] left First point.
   * @param[in] right Second point.
   * @return Whether dimensions and coordinates match exactly.
   */
  static bool samePoint( Eigen::Ref< const Eigen::VectorXd > left,
                         Eigen::Ref< const Eigen::VectorXd > right );

  /** Copied GPS, ranking-and-selection, and stopping settings. */
  RinottRankingSelectionGPSHyperparameters hyperparameters_;
  /** Owned stochastic convergence strategy. */
  std::unique_ptr< StochasticGPSStoppingStrategy > stopping_strategy_;
  /** Current ranking-and-selection overlay phase. */
  RankingSelectionPhase ranking_selection_phase_ =
    RankingSelectionPhase::Delegating;
  /** GPS search or poll waiting phase wrapped by the active selection. */
  Phase pending_gps_phase_ = Phase::WaitingSearch;

  /** Deduplicated trial alternatives followed by the incumbent. */
  std::vector< RankingSelectionCandidate > candidates_;
  /** Candidate index for each requested physical evaluation. */
  std::vector< std::size_t > pending_candidate_indices_;
  /** Index of the incumbent, always the final candidates_ entry. */
  std::size_t incumbent_index_ = 0;
  /** Complete aggregate selected as the current stochastic incumbent. */
  CandidateEvaluation incumbent_evaluation_;

  /** Current \f$\alpha_r\f$, decayed after every selection invocation. */
  double significance_;
  /** Current \f$\delta_r\f$, decayed after every selection invocation. */
  double indifference_;
  /** Bessel-corrected deviation of the latest selected incumbent. */
  std::optional< double > incumbent_standard_deviation_;
  /** Number of completed ranking-and-selection invocations. */
  std::size_t ranking_selection_call_ = 0;
  /** Prevents repeated multiple-inner-sample diagnostics. */
  bool multiple_samples_warning_emitted_ = false;
};

/** @} */

#endif // !OPTIMIZER_RINOTT_RANKING_SELECTION_GPS_METHOD_H
