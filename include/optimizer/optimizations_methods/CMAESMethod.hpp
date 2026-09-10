#ifndef CMA_ES_METHOD_H
#define CMA_ES_METHOD_H

/**
 * @defgroup optimization_methods_api Optimization Methods
 * @brief Concrete optimization state machines, GPS and Nelder--Mead families, cooling, and surrogates.
 * @{
 */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <random>
#include <vector>
#include "MethodHyperparameters.hpp"
#include "OptimizerMethod.hpp"

class OptimizerFactory;

/**
 * @brief Adapts a multivariate Gaussian search distribution with CMA-ES.
 *
 * A generation samples
 * \f[
 * z_i\sim\mathcal N(0,I),\qquad y_i=B_kD_kz_i,\qquad
 * x_i^k=m_k+\sigma_k y_i,
 * \f]
 * where \f$C_k=B_kD_k^2B_k^T\f$. With \f$\lambda\f$ equal to population_size,
 * \f$\mu=\lfloor\lambda/2\rfloor\f$ parents receive
 * \f[
 * w'_i=\log((\lambda+1)/2)-\log(i),\quad
 * w_i=\frac{w'_i}{\sum_jw'_j},\quad
 * \mu_{eff}=\frac{1}{\sum_iw_i^2},\quad i=1,\ldots,\mu.
 * \f]
 * Successful candidates are sorted by minimization mean. Fewer than \f$\mu\f$
 * successes leave distribution state unchanged while still completing the
 * optimizer's logical batch.
 *
 * @see optimization_method_states_chapter
 */
class CMAESMethod : public OptimizerMethod
{
  public:
  /**
   * @brief Samples and stores one full population from the current distribution.
   * @return Exactly population_size candidate vectors in standard-normal draw order.
   */
  std::vector< Eigen::VectorXd > ask() override;
  /**
   * @brief Restores the construction-time mean, identity covariance, initial sigma, paths, and random seed.
   * @note Construction samples the initialization strategy once. Reset reuses initial_mean_.
   */
  void reset() override;

  protected:
  /**
   * @brief Recombines successful parents and performs one ordered CMA update.
   *
   * The order is mean, step-size path, sigma, h_sigma gate, covariance path,
   * covariance matrix, then eigendecomposition.
   *
   * @param[in] evaluations One result for every most recently sampled candidate.
   * @return A completed logical generation.
   * @throws std::invalid_argument If population cardinality or a candidate dimension is wrong.
   * @throws ConfigurationEvaluationError If the updated covariance is not finite positive definite.
   */
  OptimizerMethodUpdate tellImpl( const std::vector< CandidateEvaluation > &evaluations ) override;

  private:
  /** Allows the factory to invoke private assembly constructors. */
  friend class OptimizerFactory;

  /**
   * @brief Constructs CMA-ES with default hyperparameters.
   * @param[in] number_of_parameters Search dimension.
   * @param[in] initial_parameters_strategy Non-owning strategy sampled once for the mean.
   */
  CMAESMethod( std::size_t number_of_parameters, InitialParametersStrategy *initial_parameters_strategy );

  /**
   * @brief Validates settings, samples the initial mean, and initializes strategy constants.
   * @param[in] hyperparameters Initial sigma, population size, and random seed.
   * @param[in] number_of_parameters Positive search dimension.
   * @param[in] initial_parameters_strategy Non-null, non-owning mean strategy.
   * @throws std::invalid_argument If configuration or generated mean dimension/finiteness is invalid.
   */
  CMAESMethod( const CMAESHyperparameters &hyperparameters, std::size_t number_of_parameters,
               InitialParametersStrategy *initial_parameters_strategy );

  /** @brief Validates strategy pointer, dimension, population size, and initial sigma. */
  void validateConfiguration() const;
  /**
   * @brief Computes recombination weights and dimension-dependent learning coefficients.
   *
   * With dimension \f$n\f$ and `effective_selection_mass_` equal to
   * \f$\mu_{eff}\f$, the stored coefficients are
   * \f{align}{
   * c_m &= 1,\\
   * c_\sigma &= \frac{\mu_{eff}+2}{n+\mu_{eff}+5},\\
   * d_\sigma &= 1+2\max\!\left(0,
   *   \sqrt{\frac{\mu_{eff}-1}{n+1}}-1\right)+c_\sigma,\\
   * c_c &= \frac{4+\mu_{eff}/n}{n+4+2\mu_{eff}/n},\\
   * c_1 &= \frac{2}{(n+1.3)^2+\mu_{eff}},\\
   * c_\mu &= \min\!\left(1-c_1,
   *   \frac{2(0.25+\mu_{eff}+1/\mu_{eff}-2)}{(n+2)^2+\mu_{eff}}\right),\\
   * \chi_n &= \sqrt n\left(1-\frac{1}{4n}+\frac{1}{21n^2}\right).
   * \f}
   * These values populate `mean_learning_rate_`, path learning rates,
   * `step_size_damping_`, rank learning rates, and `expected_normal_norm_`.
   */
  void initializeStrategyParameters();
  /** @brief Restores mutable distribution, paths, eigensystem, buffers, seed, and generation zero. */
  void initializeState();

  /** @brief Fills lambda vectors with independent standard-normal components. */
  void sampleStandardNormalVectors();
  /** @brief Computes each correlated step \f$y_i=B_kD_kz_i\f$. */
  void transformStandardNormalVectors();
  /** @brief Computes each candidate \f$x_i^k=m_k+\sigma_k y_i\f$. */
  void constructPopulation();

  /**
   * @brief Converts candidate coordinates to old-distribution steps and sorts successes.
   * @param[in] evaluations Full population feedback in ask() order.
   * @return Successful copies sorted by increasing mean, with parameters replaced by
   * \f$y_{i:\lambda}^k=(x_{i:\lambda}^k-m_k)/\sigma_k\f$.
   */
  std::vector< CandidateEvaluation >
  normalizeAndSortSuccessfulEvaluations( const std::vector< CandidateEvaluation > &evaluations ) const;
  /**
   * @brief Computes the selected weighted step \f$y_w^k=\sum_{i=1}^{\mu}w_i y_{i:\lambda}^k\f$.
   * @param[in] sorted_candidates Successful normalized candidates from best to worst.
   * @return Weighted old-distribution step.
   */
  Eigen::VectorXd calculateWeightedMeanStep( const std::vector< CandidateEvaluation > &sorted_candidates ) const;
  /** @brief Applies \f$m_{k+1}=m_k+c_m\sigma_k y_w^k\f$. @param[in] weighted_mean_step \f$y_w^k\f$. */
  void updateMean( const Eigen::VectorXd &weighted_mean_step );
  /**
   * @brief Updates the whitened step-size evolution path.
   *
   * Whitening uses \f$C_k^{-1/2}y_w^k=B_kD_k^{-1}B_k^Ty_w^k\f$, then
   * \f$p_{\sigma,k+1}=(1-c_\sigma)p_{\sigma,k}+
   * \sqrt{c_\sigma(2-c_\sigma)\mu_{eff}}C_k^{-1/2}y_w^k\f$.
   *
   * @param[in] weighted_mean_step \f$y_w^k\f$ under the old distribution.
   */
  void updateStepSizePath( const Eigen::VectorXd &weighted_mean_step );
  /**
   * @brief Applies cumulative step-size adaptation.
   *
   * \f$\sigma_{k+1}=\sigma_k\exp((c_\sigma/d_\sigma)
   * (\lVert p_{\sigma,k+1}\rVert/\chi_n-1))\f$.
   */
  void updateGlobalStepSize();
  /**
   * @brief Computes the binary covariance-path gate.
   *
   * For `generation_` equal to \f$k\f$ before the current update, the gate is
   * one exactly when
   * \f[
   * \frac{\lVert p_{\sigma,k+1}\rVert}
   * {\sqrt{1-(1-c_\sigma)^{2(k+1)}}}
   * < \left(1.4+\frac{2}{n+1}\right)\chi_n.
   * \f]
   *
   * @return One when the strict inequality holds; zero otherwise.
   */
  double calculateHSigma() const;
  /**
   * @brief Updates \f$p_{c,k+1}=(1-c_c)p_{c,k}+h_\sigma
   * \sqrt{c_c(2-c_c)\mu_{eff}}y_w^k\f$.
   * @param[in] weighted_mean_step Selected weighted step.
   * @param[in] h_sigma Binary gate returned by calculateHSigma().
   */
  void updateCovariancePath( const Eigen::VectorXd &weighted_mean_step, double h_sigma );
  /**
   * @brief Applies covariance retention, rank-one, and rank-mu terms.
   *
   * With \f$a_C=1-c_1-c_\mu+(1-h_\sigma)c_1c_c(2-c_c)\f$, the update is
   * \f$C_{k+1}=a_C C_k+c_1p_{c,k+1}p_{c,k+1}^T
   * +c_\mu\sum_iw_i y_{i:\lambda}^k(y_{i:\lambda}^k)^T\f$.
   *
   * @param[in] sorted_candidates Successful normalized candidates in rank order.
   * @param[in] h_sigma Binary covariance-path gate.
   */
  void updateCovarianceMatrix( const std::vector< CandidateEvaluation > &sorted_candidates, double h_sigma );
  /**
   * @brief Symmetrizes \f$C_{k+1}\f$ and refreshes \f$B_{k+1}\f$ and diagonal \f$D_{k+1}\f$.
   * @throws ConfigurationEvaluationError If decomposition fails or an eigenvalue is non-finite or non-positive.
   */
  void updateEigendecomposition();

  /** Copied initial sigma, population size, and seed. */
  CMAESHyperparameters hyperparameters_;

  std::size_t dimension_;                                  ///< \f$n\f$: optimization parameter count.
  std::size_t population_size_;                            ///< \f$\lambda\f$: candidates per generation.
  std::size_t selected_parent_count_;                      ///< \f$\mu=\lfloor\lambda/2\rfloor\f$: recombined parents.
  Eigen::VectorXd recombination_weights_;                  ///< Positive normalized \f$w_i\f$ in rank order.
  double effective_selection_mass_;                        ///< \f$\mu_{eff}=1/\sum_iw_i^2\f$.

  double mean_learning_rate_;                              ///< Mean rate \f$c_m=1\f$.
  double step_size_path_learning_rate_;                    ///< Step-size path rate \f$c_\sigma\f$.
  double step_size_damping_;                               ///< Global step-size damping \f$d_\sigma\f$.
  double covariance_path_learning_rate_;                   ///< Covariance path rate \f$c_c\f$.
  double rank_one_learning_rate_;                          ///< Rank-one covariance rate \f$c_1\f$.
  double rank_mu_learning_rate_;                           ///< Rank-mu covariance rate \f$c_\mu\f$.
  double expected_normal_norm_;                            ///< Approximation \f$\chi_n=E\lVert\mathcal N(0,I)\rVert\f$.

  Eigen::VectorXd initial_mean_;                           ///< Construction-time distribution mean \f$m_0\f$.
  Eigen::VectorXd mean_;                                   ///< Current distribution mean \f$m_k\f$.
  double global_step_size_;                                ///< Current global step size \f$\sigma_k\f$.
  Eigen::MatrixXd covariance_matrix_;                      ///< Covariance shape \f$C_k\f$.
  Eigen::VectorXd step_size_path_;                         ///< Step-size evolution path \f$p_{\sigma,k}\f$.
  Eigen::VectorXd covariance_path_;                        ///< Covariance evolution path \f$p_{c,k}\f$.

  Eigen::MatrixXd eigenvectors_;                           ///< Orthogonal covariance eigenvectors \f$B_k\f$.
  Eigen::VectorXd axis_scaling_;                           ///< Diagonal square-root eigenvalue matrix \f$D_k\f$.

  std::vector< Eigen::VectorXd > standard_normal_vectors_; ///< Current \f$z_i\sim\mathcal N(0,I)\f$ in draw order.
  std::vector< Eigen::VectorXd > transformed_vectors_;     ///< Current \f$y_i=B_kD_kz_i\f$ in matching order.
  std::vector< Eigen::VectorXd > population_;              ///< Current \f$x_i^k=m_k+\sigma_k y_i\f$ in matching order.

  /** Random engine seeded from hyperparameters_. */
  std::mt19937_64 generator_;
  /** Reusable scalar standard-normal distribution. */
  std::normal_distribution< double > standard_normal_distribution_;

  /** Number of generations that performed a distribution update. */
  std::size_t generation_ = 0;

  std::size_t eigendecomposition_period_;
};

/** @} */

#endif // !CMA_ES_METHOD_H
