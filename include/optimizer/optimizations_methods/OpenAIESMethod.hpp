#ifndef OPENAI_ES_METHOD_H
#define OPENAI_ES_METHOD_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <optional>
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
   * @brief Samples one complete antithetic population and its unperturbed mean.
   * @return population_size perturbations followed by the current mean.
   * @throws std::logic_error If prior feedback is pending.
   * @throws std::invalid_argument If the first generated mean is empty.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /** @brief Clears mean and noise buffers, restores the random seed, and returns to initialization. */
  void reset() override;

  /**
   * @brief Returns the current OpenAI-ES search-distribution center.
   *
   * After a complete tell() this is the updated center that the next
   * generation would perturb, not the best perturbation from the last batch.
   */
  const Eigen::VectorXd &currentCenter() const;

  /** @brief Gaussian perturbation scale used by the next ask(). */
  double currentNoiseScale() const noexcept { return current_noise_scale_; }
  /** @brief Offspring fraction that beat the parent in the previous generation. */
  std::optional< double > lastMutationSuccessRate() const noexcept
  {
    return last_mutation_success_rate_;
  }

  /** @brief Complete ES-center state required to continue an Adam or AdamW search. */
  struct CenterState
  {
    Eigen::VectorXd parameters;
    Eigen::VectorXd adam_first_moment;
    Eigen::VectorXd adam_second_moment;
    std::size_t adam_step = 0;
    double noise_scale = 0.0;
  };

  /** @brief Captures the current center together with its Adam/AdamW momentum state. */
  CenterState centerState() const;
  /** @brief Supplies a complete state that will replace the ordinary initialization on first ask(). */
  void setInitialCenterState( const CenterState &state );
  /** @brief Restores a center and its corresponding Adam/AdamW momentum state. */
  void restoreCenterState( const CenterState &state );

  protected:
  /**
   * @brief Computes utilities and updates the current distribution mean.
   * @param[in] evaluations Full population feedback in antithetic request order.
   * @return A completed logical generation.
   * @throws std::logic_error If no population is pending.
   * @throws std::invalid_argument If feedback cardinality differs from population_size.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;
  void updateBestCandidate( const std::vector< CandidateEvaluation > &evaluations ) override;

  private:
  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /** @brief Constructs the method with default settings. @param[in] initial_parameters_strategy Non-owning mean
   * strategy. */
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
  /** @brief Validates a serialized or in-memory continuation state. */
  void validateCenterState( const CenterState &state, Eigen::Index expected_parameter_count ) const;
  /** @brief Samples one vector of independent standard-normal components. @return Noise matching current mean
   * dimension. */
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
  /** @brief Updates the mean with the configured optimizer. */
  void updateParameters( const Eigen::VectorXd &search_direction );
  /** @brief Applies a stochastic-gradient-descent mean update. */
  void updateWithSGD( const Eigen::VectorXd &search_direction );
  /** @brief Computes the bias-corrected Adam/AdamW search step and updates their shared moments. */
  Eigen::VectorXd adamSearchStep( const Eigen::VectorXd &search_direction );
  /** @brief Applies an Adam mean update. */
  void updateWithAdam( const Eigen::VectorXd &search_direction );
  /** @brief Applies an AdamW mean update with decoupled weight decay. */
  void updateWithAdamW( const Eigen::VectorXd &search_direction );
  /** @brief Applies the bounded one-fifth success rule to the Gaussian scale. */
  void adaptNoiseScale( const std::vector< CandidateEvaluation > &evaluations );

  /** Copied population, update, fitness, and seed settings. */
  OpenAIESHyperparameters hyperparameters_;
  /** Gaussian perturbation scale, initialized from hyperparameters_.noise_scale. */
  double current_noise_scale_;
  /** Last population fraction that beat its unperturbed parent center. */
  std::optional< double > last_mutation_success_rate_;
  /** Random engine used for Gaussian directions. */
  std::mt19937_64 generator_;
  /** Reusable scalar standard-normal distribution. */
  std::normal_distribution< double > standard_normal_distribution_{ 0.0, 1.0 };
  /** Current search-distribution mean \f$\theta\f$. */
  Eigen::VectorXd current_parameters_;
  /** Adam/AdamW first-moment estimate. */
  Eigen::VectorXd adam_first_moment_;
  /** Adam/AdamW second-moment estimate. */
  Eigen::VectorXd adam_second_moment_;
  /** Number of Adam or AdamW updates since initialization or reset. */
  std::size_t adam_step_ = 0;
  /** Complete continuation state consumed by the first ask() after reset. */
  std::optional< CenterState > pending_initial_center_state_;
  /** One \f$\varepsilon_i\f$ per adjacent antithetic pair. */
  std::vector< Eigen::VectorXd > sampled_noises_;
  /** Whether current_parameters_ has been generated. */
  bool initialized_ = false;
  /** Whether the current antithetic population awaits feedback. */
  bool waiting_for_evaluations_ = false;
};

/** @} */

#endif // !OPENAI_ES_METHOD_H
