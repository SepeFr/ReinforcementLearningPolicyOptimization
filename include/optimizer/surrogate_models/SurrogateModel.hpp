#ifndef SURROGATE_MODEL_H
#define SURROGATE_MODEL_H

/** @addtogroup optimization_methods_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <vector>
#include "CandidateEvaluation.hpp"
#include "surrogate_models/SurrogateModelConfiguration.hpp"

/** @brief Interface for a local minimization-value predictor. */
class SurrogateModel
{
  public:
  /** @brief Enables destruction through the model interface. */
  virtual ~SurrogateModel() = default;

  /**
   * @brief Fits successful local history in coordinates \f$\widetilde{x}=(x-x_k)/\rho_k\f$.
   * @param[in] history Candidate observations available to the search strategy.
   * @param[in] center Current mesh center \f$x_k\f$.
   * @param[in] scale Positive mesh scale \f$\rho_k\f$.
   * @return `true` when sufficient independent data produced a finite model.
   */
  virtual bool fit( const std::vector< CandidateEvaluation > &history, Eigen::Ref< const Eigen::VectorXd > center,
                    double scale ) = 0;

  /**
   * @brief Predicts a minimization value using the latest successful fit.
   * @param[in] parameters Candidate in original problem coordinates.
   * @return Predicted objective mean.
   */
  virtual double predict( Eigen::Ref< const Eigen::VectorXd > parameters ) const = 0;
  /** @brief Clears fitted state. */
  virtual void reset() = 0;
};

/**
 * @brief Fits a regularized full quadratic model to local successful history.
 *
 * In scaled coordinates \f$\widetilde{x}=(x-x_k)/\rho_k\f$, prediction has the form
 * \f[
 * s_k(x)=c+g^T\widetilde{x}+\tfrac12\widetilde{x}^TH\widetilde{x}.
 * \f]
 * Curvature regularization penalizes \f$\lVert H\rVert_F^2\f$; off-diagonal
 * coefficients therefore receive a \f$\sqrt{2}\f$ row weight. A dimension
 * \f$n\f$ needs at least \f$1+n+n(n+1)/2\f$ local observations and a full-rank
 * augmented design.
 */
class LocalQuadraticSurrogate : public SurrogateModel
{
  public:
  /**
   * @brief Stores validated local-radius and curvature settings.
   * @param[in] configuration Regression settings.
   * @throws std::invalid_argument If radius multiplier is non-positive or
   * non-finite, or curvature regularization is negative or non-finite.
   */
  explicit LocalQuadraticSurrogate( LocalQuadraticSurrogateConfiguration configuration );

  /**
   * @brief Fits successful, dimension-matching observations inside the local radius.
   * @param[in] history Candidate observations in arrival order.
   * @param[in] center Finite local origin.
   * @param[in] scale Finite positive coordinate scale.
   * @return `false` for invalid center/scale, insufficient observations, rank
   * deficiency, or non-finite fitted coefficients; `true` after storing a model.
   */
  bool fit( const std::vector< CandidateEvaluation > &history, Eigen::Ref< const Eigen::VectorXd > center,
            double scale ) override;

  /**
   * @brief Evaluates the stored quadratic in scaled coordinates.
   * @param[in] parameters Coordinates matching the fitted center dimension.
   * @return Predicted minimization value.
   * @throws std::logic_error If no successful fit is stored.
   * @throws std::invalid_argument If the dimension differs from the fitted model.
   */
  double predict( Eigen::Ref< const Eigen::VectorXd > parameters ) const override;
  /** @brief Clears the fitted center and coefficients and restores scale to one. */
  void reset() override;

  private:
  /**
   * @brief Builds intercept, linear, diagonal-quadratic, then mixed-quadratic features.
   * @param[in] scaled_parameters Local coordinate vector \f$\widetilde{x}\f$.
   * @return Feature vector in coefficient storage order.
   */
  Eigen::VectorXd makeFeatures( Eigen::Ref< const Eigen::VectorXd > scaled_parameters ) const;
  /**
   * @brief Builds rows that apply Frobenius regularization only to curvature.
   * @param[in] parameter_count Fitted coordinate dimension.
   * @return Curvature-row matrix aligned with makeFeatures() coefficients.
   */
  Eigen::MatrixXd makeRegularizationMatrix( Eigen::Index parameter_count ) const;

  /** Copied curvature and local-radius configuration. */
  LocalQuadraticSurrogateConfiguration configuration_;
  /** Center \f$x_k\f$ from the latest successful fit. */
  Eigen::VectorXd center_;
  /** Scale \f$\rho_k\f$ from the latest successful fit. */
  double scale_ = 1.0;
  /** Fitted intercept, gradient, diagonal curvature, and mixed curvature coefficients. */
  Eigen::VectorXd coefficients_;
};

/**
 * @brief Fits a local radial-basis model with an affine polynomial tail.
 *
 * For scaled coordinates, prediction is
 * \f$s_k(x)=p(\widetilde{x})+\sum_i\lambda_i
 * \phi(\lVert\widetilde{x}-\widetilde{x}_i\rVert)\f$.
 * fit() solves
 * \f[
 * \begin{bmatrix}\Phi+\lambda_R I&P\\P^T&0\end{bmatrix}
 * \begin{bmatrix}\lambda\\\beta\end{bmatrix}=
 * \begin{bmatrix}F\\0\end{bmatrix}.
 * \f]
 * The coreset begins with \f$n+1\f$ affinely independent points and is filled
 * to its configured cap by maximin distance selection.
 */
class RBFSurrogate : public SurrogateModel
{
  public:
  /**
   * @brief Stores validated kernel, regularization, radius, and coreset settings.
   * @param[in] configuration RBF settings.
   * @throws std::invalid_argument If a numeric constraint fails, coreset size
   * is zero, or the kernel enum is unsupported.
   */
  explicit RBFSurrogate( RBFSurrogateConfiguration configuration );

  /**
   * @brief Selects a local coreset and solves the RBF block system.
   * @param[in] history Candidate observations in arrival order.
   * @param[in] center Finite local origin.
   * @param[in] scale Finite positive coordinate scale.
   * @return `false` for invalid center/scale, insufficient affine data, a
   * singular system, or non-finite solution; `true` after storing a model.
   */
  bool fit( const std::vector< CandidateEvaluation > &history, Eigen::Ref< const Eigen::VectorXd > center,
            double scale ) override;

  /**
   * @brief Evaluates radial and affine terms of the stored model.
   * @param[in] parameters Coordinates matching the fitted center dimension.
   * @return Predicted minimization value.
   * @throws std::logic_error If no successful fit is stored.
   * @throws std::invalid_argument If the dimension differs from the fitted model.
   */
  double predict( Eigen::Ref< const Eigen::VectorXd > parameters ) const override;
  /** @brief Clears fitted centers and coefficients and restores scale to one. */
  void reset() override;

  private:
  /**
   * @brief Greedily selects an affine core by modified Gram--Schmidt residuals.
   * @param[in] candidates Successful local candidates.
   * @param[in] center Current local origin used to choose the closest anchor.
   * @param[in] scale Positive local coordinate scale.
   * @return Selected indices beginning with the closest anchor; fewer than
   * `dimension + 1` indicates affine dependence.
   */
  std::vector< std::size_t >
  selectAffinelyIndependentCandidates( const std::vector< CandidateEvaluation > &candidates,
                                       Eigen::Ref< const Eigen::VectorXd > center, double scale ) const;

  /**
   * @brief Extends an affine core with farthest-from-selected candidates.
   * @param[in] candidates Local candidates.
   * @param[in,out] selected_indices Existing core followed by maximin additions.
   * @param[in] target_count Requested cap, limited by candidate cardinality.
   */
  void completeCoresetWithMaximin( const std::vector< CandidateEvaluation > &candidates,
                                   std::vector< std::size_t > &selected_indices,
                                   std::size_t target_count ) const;

  /**
   * @brief Evaluates the configured radial kernel.
   * @param[in] radius Non-negative scaled Euclidean distance.
   * @return \f$r^3\f$ or \f$r\f$.
   * @throws std::logic_error If the stored kernel enum is unsupported.
   */
  double radialBasisValue( double radius ) const;

  /** Copied kernel, regularization, radius, and coreset settings. */
  RBFSurrogateConfiguration configuration_;
  /** Center \f$x_k\f$ from the latest successful fit. */
  Eigen::VectorXd center_;
  /** Scale \f$\rho_k\f$ from the latest successful fit. */
  double scale_ = 1.0;
  /** Selected coreset centers represented as \f$\widetilde{x}_i=(x_i-x_k)/\rho_k\f$. */
  std::vector< Eigen::VectorXd > scaled_centers_;
  /** Fitted \f$\lambda_i\f$ radial coefficients. */
  Eigen::VectorXd radial_coefficients_;
  /** Fitted affine coefficients `[a_0, a]`. */
  Eigen::VectorXd polynomial_coefficients_;
};

/** @} */

#endif // !SURROGATE_MODEL_H
