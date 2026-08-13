#ifndef OPTIMIZER_SIMULATED_ANNEALING_METHOD_H
#define OPTIMIZER_SIMULATED_ANNEALING_METHOD_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <eigen3/Eigen/Core>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include "optimizations_methods/simulated_annealing/CoolingFunction.hpp"
#include "MethodHyperparameters.hpp"
#include "NeighborhoodStrategy.hpp"
#include "OptimizerMethod.hpp"

class OptimizerFactory;

/**
 * @brief Performs single-proposal Metropolis simulated annealing.
 *
 * A lower-mean proposal is accepted directly. Other successful proposals with
 * energy increase \f$\Delta=f(x')-f(x)\f$ are accepted with probability
 * \f$\exp(-\Delta/T_k)\f$. Time starts at one and advances after every local
 * proposal, including a failed evaluation.
 *
 * @see optimization_method_states_chapter
 */
class SimulatedAnnealingMethod : public OptimizerMethod
{
  public:
  /**
   * @brief Produces the initial point or one scaled local proposal.
   * @return Batch containing exactly one candidate.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Clears the current point, resets time to one, and reseeds random components. */
  void reset() override;

  /**
   * @brief Compares the current scheduled temperature with the strict threshold.
   * @return Converged when `temperature(time_) < minimal_temperature`, otherwise InProgress.
   */
  ConvergenceStatus convergenceStatus() const override;

  protected:
  /**
   * @brief Initializes the current point or applies one Metropolis transition.
   * @param[in] evaluations Feedback for the single candidate returned by ask().
   * @return A completed logical iteration.
   * @throws std::out_of_range If initialization feedback is empty.
   * @throws ConfigurationEvaluationError If the initial candidate failed.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;

  private:
  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /**
   * @brief Constructs a default Gaussian and geometric-cooling method.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   */
  explicit SimulatedAnnealingMethod( InitialParametersStrategy *initial_parameters_strategy ) :
      SimulatedAnnealingMethod(
        SimulatedAnnealingHyperparameters{}, std::make_unique< GaussianNeighborhood >(),
        std::make_unique< GeometricCooling >( SimulatedAnnealingHyperparameters{}.initial_temperature,
                                              GeometricCoolingConfiguration{} ),
        initial_parameters_strategy, 0 )
  {
  }

  /**
   * @brief Takes ownership of proposal and cooling strategies.
   * @param[in] hyperparameters Scale and convergence settings copied into the method.
   * @param[in] neighborhood Non-null owned proposal strategy.
   * @param[in] cooling_function Non-null owned schedule.
   * @param[in] initial_parameters_strategy Non-owning initialization strategy.
   * @param[in] random_seed Seed used by the Metropolis acceptance draw.
   */
  SimulatedAnnealingMethod( const SimulatedAnnealingHyperparameters &hyperparameters,
                            std::unique_ptr< NeighborhoodStrategy > neighborhood,
                            std::unique_ptr< CoolingFunction > cooling_function,
                            InitialParametersStrategy *initial_parameters_strategy,
                            std::uint64_t random_seed ) :
      OptimizerMethod( initial_parameters_strategy ), neighborhood_( std::move( neighborhood ) ),
      cooling_function_( std::move( cooling_function ) ), hyperparameters_( hyperparameters ),
      random_seed_( random_seed ), generator_( random_seed )
  {
    assert( neighborhood_ && "SimulatedAnnealingMethod: neighborhood cannot be null" );
    assert( cooling_function_ && "SimulatedAnnealingMethod: cooling function cannot be null" );
  }

  /** Owned proposal displacement strategy. */
  std::unique_ptr< NeighborhoodStrategy > neighborhood_;
  /** Owned temperature schedule. */
  std::unique_ptr< CoolingFunction > cooling_function_;
  /** Copied proposal scale and temperature thresholds. */
  SimulatedAnnealingHyperparameters hyperparameters_;
  /** Construction seed retained for reset(). */
  std::uint64_t random_seed_;
  /** Random engine used by the Metropolis draw. */
  std::mt19937_64 generator_;
  /** Uniform distribution on \f$[0,1)\f$ used for acceptance. */
  std::uniform_real_distribution< double > unit_distribution_{ 0.0, 1.0 };
  /** Current cooling time; starts at one. */
  std::size_t time_ = 1;
  /** Accepted current point and evaluation. */
  CandidateEvaluation current_candidate_;
  /** Distinguishes initialization from proposal iterations. */
  bool has_current_candidate_ = false;
};

/** @} */

#endif // !OPTIMIZER_SIMULATED_ANNEALING_METHOD_H
