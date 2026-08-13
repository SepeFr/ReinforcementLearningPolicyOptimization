#ifndef OPTIMIZER_STOCHASTIC_HEURISTIC_NELDER_MEAD_METHOD_H
#define OPTIMIZER_STOCHASTIC_HEURISTIC_NELDER_MEAD_METHOD_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <vector>
#include "MethodHyperparameters.hpp"
#include "optimizations_methods/nelder_mead/NelderMeadMethod.hpp"

class OptimizerFactory;

/**
 * @brief Adds targeted reevaluations to the deterministic simplex state machine.
 *
 * Optional overlays refresh a best vertex after it remains unchanged for
 * vertexCount() complete simplexes, confirm values involved in contraction,
 * and refresh the shrink anchor. Successful refreshes replace stored
 * observations directly.
 *
 * @see optimization_method_states_chapter
 */
class StochasticHeuristicNelderMeadMethod final : public NelderMeadMethod
{
  public:
  /**
   * @brief Produces an active reevaluation batch or delegates to NelderMeadMethod::ask().
   * @return Parameters required by the heuristic or base phase.
   * @throws std::logic_error If heuristic feedback is pending.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Resets the base simplex and every heuristic overlay flag and counter. */
  void reset() override;

  protected:
  /**
   * @brief Routes feedback to the active heuristic or base phase.
   * @param[in] evaluations Ordered feedback for the preceding ask().
   * @return Whether a logical base simplex iteration completed.
   * @throws std::invalid_argument If a heuristic batch has the wrong cardinality.
   * @throws std::logic_error If no feedback is pending.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;

  private:
  /** @brief Overlay phases that temporarily intercept the base simplex flow. */
  enum class HeuristicPhase
  {
    Delegating,                     ///< Base Nelder--Mead handles ask and tell.
    ReevaluateStagnantBest,         ///< Ready to request the unchanged best vertex.
    WaitingStagnantBest,            ///< Awaiting the stagnant-best reevaluation.
    ConfirmContraction,             ///< Ready to request contraction comparison values.
    WaitingContractionConfirmation, ///< Awaiting contraction confirmation feedback.
    RefreshBestBeforeShrink,        ///< Ready to request the current shrink anchor.
    WaitingBestBeforeShrink         ///< Awaiting the pre-shrink best reevaluation.
  };

  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /**
   * @brief Constructs the stochastic heuristic with default settings.
   * @param[in] number_of_parameters Problem dimension.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   */
  StochasticHeuristicNelderMeadMethod(
    std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Constructs the base simplex and copies heuristic controls.
   * @param[in] hyperparameters Base and stochastic heuristic settings.
   * @param[in] number_of_parameters Problem dimension.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   * @note A shrink coefficient equal to `0.5` emits an operational warning to stderr.
   */
  StochasticHeuristicNelderMeadMethod(
    const StochasticHeuristicNelderMeadHyperparameters &hyperparameters,
    std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy );

  /** @brief Updates the unchanged-best cycle count and schedules reevaluation when due. */
  void updateBestStagnation();
  /**
   * @brief Captures contraction context and builds its confirmation batch.
   * @return Reflection plus worst for inside contraction; reflection,
   * second-worst, and worst for outside contraction.
   */
  std::vector< Eigen::VectorXd > contractionConfirmationCandidates();
  /** @brief Applies one stagnant-best refresh. @param[in] evaluations Single requested result. */
  void handleStagnantBestEvaluation( const std::vector< CandidateEvaluation > &evaluations );
  /** @brief Applies refreshed contraction comparison values. @param[in] evaluations Two or three ordered results. */
  void handleContractionConfirmation( const std::vector< CandidateEvaluation > &evaluations );
  /** @brief Refreshes and reorders the best before a shrink decision. @param[in] evaluations Single requested result. */
  void handleBestBeforeShrinkEvaluation( const std::vector< CandidateEvaluation > &evaluations );
  /**
   * @brief Tests exact coordinate identity.
   * @param[in] left First point.
   * @param[in] right Second point.
   * @return Whether dimensions and all coordinate values match exactly.
   */
  static bool samePoint( Eigen::Ref< const Eigen::VectorXd > left,
                         Eigen::Ref< const Eigen::VectorXd > right );

  /** Copied base and heuristic configuration. */
  StochasticHeuristicNelderMeadHyperparameters hyperparameters_;
  /** Current overlay phase. */
  HeuristicPhase heuristic_phase_ = HeuristicPhase::Delegating;

  /** Coordinates of the best vertex from the previous complete simplex. */
  Eigen::VectorXd last_best_parameters_;
  /** Worst coordinates captured before contraction confirmation. */
  Eigen::VectorXd contraction_worst_parameters_;
  /** Base contraction phase that produced the confirmation request. */
  Phase pending_contraction_phase_ = Phase::OutsideContraction;
  /** Whether last_best_parameters_ has been initialized. */
  bool has_last_best_ = false;
  /** Allows one base contraction operation after a valid confirmation. */
  bool contraction_confirmation_complete_ = false;
  /** Allows one base shrink after its anchor remains best. */
  bool shrink_refresh_complete_ = false;
  /** Consecutive complete simplexes retaining exactly the same best coordinates. */
  std::size_t unchanged_best_cycles_ = 0;
};

/** @} */

#endif // !OPTIMIZER_STOCHASTIC_HEURISTIC_NELDER_MEAD_METHOD_H
