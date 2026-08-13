#ifndef OPENAI_ES_METHOD_H
#define OPENAI_ES_METHOD_H

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
 * @brief Estimates a smoothed-objective direction from antithetic Gaussian pairs.
 *
 * For each stored \f$\varepsilon_i\sim\mathcal N(0,I)\f$, ask() emits adjacent
 * candidates \f$\theta+\sigma\varepsilon_i\f$ then
 * \f$\theta-\sigma\varepsilon_i\f$. Successful pair utilities produce
 * \f[
 * \widehat g=\frac{1}{\lambda\sigma}\sum_{i=1}^{\lambda/2}
 * (u_i^+-u_i^-)\varepsilon_i,\qquad
 * \theta\leftarrow\theta+\eta\widehat g.
 * \f]
 * Raw utility is the negated minimization mean. Centered-rank utility maps the
 * best through worst successful candidates linearly from `0.5` to `-0.5`;
 * failed candidates receive zero. A pair with either failure contributes zero.
 *
 * @see optimization_method_states_chapter
 */
class OpenAIESMethod : public OptimizerMethod
{
  public:
  /**
   * @brief Samples one complete antithetic population around the current mean.
   * @return population_size candidates ordered as adjacent positive/negative pairs.
   * @throws std::logic_error If prior feedback is pending.
   * @throws std::invalid_argument If the first generated mean is empty.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Clears mean and noise buffers, restores the random seed, and returns to initialization. */
  void reset() override;

  protected:
  /**
   * @brief Computes utilities and updates the current distribution mean.
   * @param[in] evaluations Full population feedback in antithetic request order.
   * @return A completed logical generation.
   * @throws std::logic_error If no population is pending.
   * @throws std::invalid_argument If feedback cardinality differs from population_size.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;

  private:
  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /** @brief Constructs the method with default settings. @param[in] initial_parameters_strategy Non-owning mean strategy. */
  explicit OpenAIESMethod( InitialParametersStrategy *initial_parameters_strategy );
  /**
   * @brief Stores validated settings and seeds the noise generator.
   * @param[in] hyperparameters Even population, learning rate, noise scale, rank option, and seed.
   * @param[in] initial_parameters_strategy Non-owning mean strategy.
   * @throws std::invalid_argument If population is odd or below two, or a scale is non-positive or non-finite.
   */
  OpenAIESMethod( const OpenAIESHyperparameters &hyperparameters,
                  InitialParametersStrategy *initial_parameters_strategy );

  /** @brief Generates and stores the initial nonempty mean from the strategy. */
  void initializeParameters();
  /** @brief Samples one vector of independent standard-normal components. @return Noise matching current mean dimension. */
  Eigen::VectorXd sampleStandardNoise();
  /**
   * @brief Computes configured utilities in original feedback order.
   * @param[in] evaluations Population feedback.
   * @return Centered ranks or negated successful means, with zero for failures.
   */
  std::vector< double > fitnessValues( const std::vector< CandidateEvaluation > &evaluations ) const;
  /**
   * @brief Assigns reversed centered ranks to successful candidates.
   * @param[in] evaluations Population feedback.
   * @return Utilities aligned to feedback indices; all zero with fewer than two successes.
   */
  std::vector< double > centeredRankFitness( const std::vector< CandidateEvaluation > &evaluations ) const;

  /** Copied population, update, fitness, and seed settings. */
  OpenAIESHyperparameters hyperparameters_;
  /** Random engine used for Gaussian directions. */
  std::mt19937_64 generator_;
  /** Reusable scalar standard-normal distribution. */
  std::normal_distribution< double > standard_normal_distribution_{ 0.0, 1.0 };
  /** Current search-distribution mean \f$\theta\f$. */
  Eigen::VectorXd current_parameters_;
  /** One \f$\varepsilon_i\f$ per adjacent antithetic pair. */
  std::vector< Eigen::VectorXd > sampled_noises_;
  /** Whether current_parameters_ has been generated. */
  bool initialized_ = false;
  /** Whether the current antithetic population awaits feedback. */
  bool waiting_for_evaluations_ = false;
};

/** @} */

#endif // !OPENAI_ES_METHOD_H
