#ifndef SPSA_METHOD_H
#define SPSA_METHOD_H

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
 * @brief Estimates a full gradient from two simultaneous Rademacher perturbations.
 *
 * ask() draws \f$\Delta_j\in\{-1,+1\}\f$ independently and returns
 * \f$x+c\Delta\f$ followed by \f$x-c\Delta\f$. Successful feedback gives
 * \f[
 * \widehat g_j=\frac{f(x+c\Delta)-f(x-c\Delta)}{2c\Delta_j},
 * \qquad x\leftarrow x-a\widehat g.
 * \f]
 * If either evaluation fails, the current point is retained and the logical
 * iteration still completes.
 *
 * @see optimization_method_states_chapter
 */
class SPSAMethod : public OptimizerMethod
{
  public:
  /**
   * @brief Generates the next positive and negative perturbation pair.
   * @return Exactly two candidates in positive-then-negative order.
   * @throws std::logic_error If prior pair feedback is pending.
   * @throws std::invalid_argument If the initial candidate dimension is wrong.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Clears current and perturbation vectors and restores random state. */
  void reset() override;

  protected:
  /**
   * @brief Applies a simultaneous gradient step when both pair evaluations succeed.
   * @param[in] evaluations Exactly two results in ask() order.
   * @return A completed logical iteration.
   * @throws std::logic_error If no pair is pending.
   * @throws std::invalid_argument If feedback cardinality differs from two.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;

  private:
  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /**
   * @brief Constructs SPSA with default hyperparameters.
   * @param[in] number_of_dimensions Problem parameter count.
   * @param[in] initial_parameters_strategy Non-owning initial-point strategy.
   */
  SPSAMethod( std::size_t number_of_dimensions, InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Stores validated scales and seeds the generator.
   * @param[in] hyperparameters Perturbation magnitude, step size, and seed.
   * @param[in] number_of_dimensions Problem parameter count.
   * @param[in] initial_parameters_strategy Non-owning initial-point strategy.
   * @throws std::invalid_argument If either scale is non-positive or non-finite.
   */
  SPSAMethod( const SPSAHyperparameters &hyperparameters, std::size_t number_of_dimensions,
              InitialParametersStrategy *initial_parameters_strategy );

  /** @brief Generates the current point and allocates a matching perturbation vector. */
  void initializeParameters();
  /** @brief Fills perturbation_vector_ with independent equiprobable `-1` and `+1` entries. */
  void generatePerturbation();

  /** Copied perturbation, update, and seed settings. */
  SPSAHyperparameters hyperparameters_;
  /** Required current-point dimension. */
  std::size_t number_of_dimensions_;

  /** Current optimization point \f$x\f$. */
  Eigen::VectorXd current_parameters_;
  /** Current Rademacher vector \f$\Delta\f$. */
  Eigen::VectorXd perturbation_vector_;

  /** Random engine used for Rademacher draws. */
  std::mt19937_64 generator_;
  /** Uniform distribution on \f$[0,1)\f$ mapped to signs. */
  std::uniform_real_distribution< double > unit_distribution_{ 0.0, 1.0 };

  /** Whether current_parameters_ has been generated. */
  bool initialized_ = false;
  /** Whether the current perturbation pair awaits feedback. */
  bool waiting_for_evaluations_ = false;
};

/** @} */

#endif // !SPSA_METHOD_H
