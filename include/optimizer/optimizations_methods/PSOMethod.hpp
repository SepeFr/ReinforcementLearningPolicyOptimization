#ifndef PSO_METHOD_H
#define PSO_METHOD_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <random>
#include <vector>
#include "MethodHyperparameters.hpp"
#include "OptimizerMethod.hpp"

class OptimizerFactory;

/**
 * @brief Evolves a fixed particle swarm using personal and global best positions.
 *
 * For particle \f$i\f$, each optimization step draws independent
 * \f$r_{i,1}^k,r_{i,2}^k\sim\mathcal U([0,1]^n)\f$ and applies
 * \f[
 * v_i^{k+1}=\omega v_i^k+c_1r_{i,1}^k\odot(p_i^k-x_i^k)
 * +c_2r_{i,2}^k\odot(g^k-x_i^k),\qquad
 * x_i^{k+1}=x_i^k+v_i^{k+1}.
 * \f]
 * Only successful evaluations update best positions. The initialization batch
 * establishes best state and leaves the logical iteration count unchanged.
 *
 * @see optimization_method_states_chapter
 */
class PSOMethod : public OptimizerMethod
{
  public:
  /**
   * @brief Generates the initial population or advances every particle once.
   * @return population_size positions in stable particle-index order.
   * @throws std::logic_error If prior population feedback is pending.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Clears swarm and best state, restores the random seed, and selects initialization. */
  void reset() override;

  protected:
  /**
   * @brief Updates personal and global bests for the pending population.
   * @param[in] evaluations One result per particle in stable index order.
   * @return Initialization is incomplete; optimization feedback completes one iteration.
   * @throws std::invalid_argument If feedback count differs from population_size.
   * @throws ConfigurationEvaluationError If every initial particle failed.
   * @throws std::logic_error If no feedback is pending.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;

  private:
  /** @brief Swarm operation and pending-feedback phases. */
  enum class Phase
  {
    Initialization,        ///< Ready to generate all initial positions.
    WaitingInitialization, ///< Awaiting initial population feedback.
    Optimization,          ///< Ready to update velocities and positions.
    WaitingOptimization    ///< Awaiting evolved population feedback.
  };

  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /**
   * @brief Constructs PSO with default hyperparameters.
   * @param[in] number_of_dimensions Problem parameter count.
   * @param[in] initial_parameters_strategy Non-owning particle initialization strategy.
   */
  PSOMethod( std::size_t number_of_dimensions, InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Stores settings and seeds the random engine.
   * @param[in] hyperparameters Population and velocity-update settings.
   * @param[in] number_of_dimensions Problem parameter count.
   * @param[in] initial_parameters_strategy Non-owning particle initialization strategy.
   * @throws std::invalid_argument If population is zero, a coefficient is
   * negative or non-finite, or velocity scale is non-positive or non-finite.
   */
  PSOMethod( const PSOHyperparameters &hyperparameters, std::size_t number_of_dimensions,
             InitialParametersStrategy *initial_parameters_strategy );

  /** @brief Validates population, coefficients, and velocity scale. */
  void validateHyperparameters() const;
  /** @brief Generates population_size positions and initializes matching personal-best coordinates. */
  void initializeParticles();
  /** @brief Samples every initial velocity coordinate uniformly from `[-initial_velocity_scale,+initial_velocity_scale)`. */
  void initializeVelocities();
  /** @brief Applies the coordinate-wise velocity formula, then adds each velocity to its position. */
  void updateVelocitiesAndPositions();
  /**
   * @brief Applies successful feedback to index-matched personal and global bests.
   * @param[in] evaluations Population feedback in stable particle order.
   */
  void updateBestPositions( const std::vector< CandidateEvaluation > &evaluations );

  /** Copied population and velocity-update settings. */
  PSOHyperparameters hyperparameters_;
  /** Required coordinate count for velocity storage. */
  std::size_t number_of_dimensions_;

  /** Current particle coordinates in stable index order. */
  std::vector< Eigen::VectorXd > positions_;
  /** Current velocity vectors matching positions_ indices. */
  std::vector< Eigen::VectorXd > velocities_;
  /** Lowest-mean successful coordinates observed by each particle. */
  std::vector< Eigen::VectorXd > personal_best_positions_;
  /** Matching personal best means; initialized to positive infinity. */
  std::vector< double > personal_best_values_;

  /** Lowest-mean successful position observed by the swarm. */
  Eigen::VectorXd global_best_position_;
  /** Mean associated with global_best_position_. */
  double global_best_value_ = 0.0;
  /** Whether at least one particle evaluation has succeeded. */
  bool has_global_best_ = false;

  /** Random engine for initial velocities and attraction factors. */
  std::mt19937_64 generator_;
  /** Uniform distribution on \f$[0,1)\f$. */
  std::uniform_real_distribution< double > unit_distribution_{ 0.0, 1.0 };
  /** Current operation or pending-feedback phase. */
  Phase phase_ = Phase::Initialization;
};

/** @} */

#endif // !PSO_METHOD_H
